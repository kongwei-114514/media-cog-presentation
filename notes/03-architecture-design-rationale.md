# Figure 2 与 `FastViTHD` 架构设计逻辑拆解（结合前作与源码）

本文档结合 Apple 团队的前序工作（`MobileOne` CVPR 2023 $\rightarrow$ `FastViT` ICCV 2023 $\rightarrow$ `MobileCLIP` CVPR 2024）与 `FastVLM` 官方开源代码（`apple/ml-fastvlm` 中的 `llava/model/multimodal_encoder/mobileclip/mci.py`），梳理 Figure 2 中各模块的底层设计动机与演进关系。

---

## 1. 为什么 FastVLM 原文对架构设计过程描述较少？

1. **基础组件继承自前序系列工作**：
   `FastViTHD` 并非从零设计的编码器，而是同一研究团队在 `MobileOne`、`FastViT`、`MobileCLIP` 基础上的第 4 代演进。Figure 2 中的基础模块（`Convolutional Stem`、`Patch Embed`、`RepMixer`、`ConvFFN`）均在 `FastViT` (ICCV 2023) 中完成了设计与消融；`FastVLM` 正文（Sec. 3.2 与附录 B）仅描述了**为适配高分辨率 VLM 而对 4 阶段 `FastViT / MCi2` 所做的增量改造**。
2. **Figure 2 是两节内容的综合示意图**：
   Figure 2 同时画出了 **Sec. 3.1.1 在 `FastViT` 上验证的多尺度特征融合（`(Learned) Pool` + `C`）** 与 **Sec. 3.2 提出的 5 阶段 `FastViTHD` 主干**。

---

## 2. 宏观架构问题：为什么高分辨率下传统架构会失效？

| 架构类型 | 代表模型 | 结构特点 | 高分辨率（如 $1024\times 1024$）下的结构性瓶颈 |
| :--- | :--- | :--- | :--- |
| **各向同性 Transformer (Isotropic ViT)** | ViT-L/14, SigLIP | 无下采样金字塔，全网络固定 $14\times 14$ Patch，每层均执行全局自注意力 (MHSA) | 1. 浅层在高达 $N=(1024/14)^2 \approx 5350$ 个 Token 上做 $O(N^2)$ 全局自注意力，编码耗时极高；<br>2. 最终输出数千个 Token 进入 LLM，导致 LLM Prefill 延迟爆炸。 |
| **纯卷积金字塔 (Pure ConvNet)** | ConvNeXt-L / XXL | 4 阶段纯卷积下采样（总下采样 $32\times$） | 缺乏深层自注意力的全局上下文建模能力，必须大幅增加参数量（如 ConvNeXt-XXL 达 846M）才能追平 VLM 精度，导致高分辨率推理依然缓慢。 |
| **常规 4 阶段混合架构 (4-Stage Hybrid)** | FastViT (MCi2), FastViT-Naive, ViTamin | 前 2~3 阶段卷积，第 3~4 阶段（$16\times / 32\times$ 下采样）执行自注意力 | 1. 当放大模型规模时（`FastViT-Naive`），在 $16\times$ 下采样（$1024^2$ 下仍有 $64\times 64=4096$ 个位置）上堆叠自注意力会导致编码延迟反超 ConvNeXt-L（论文 Fig. 3）；<br>2. 总下采样 $32\times$ 在 $1024^2$ 下仍输出 1024 个 Token。 |
| **5 阶段高分辨率混合架构 (5-Stage Hybrid)** | **FastViTHD (本文)** | 前 3 阶段（$4\times, 8\times, 16\times$）纯 `RepMixer`，后 2 阶段（$32\times, 64\times$）执行 `MHSA` | 自注意力仅在 $32\times$（1024 位置）和 $64\times$（256 位置）的高度浓缩张量上运行，同时将输出给 LLM 的视觉 Token 压缩至 **256 个**。 |

---

## 3. Figure 2 逐模块设计原理解析

![FastVLM Overview Architecture](../paper/figures/main_diagram.png)

### 3.1 橙色块：`Convolutional Stem`（源自 `MobileOne` & `FastViT`）
- **针对的问题**：ViT 采用 stride=14 的非重叠线性切片（Patchify），容易切断跨 Patch 边界的局部连续结构（如细小文字笔画），且缺乏归纳偏置。
- **具体实现（`mci.py` 中 `convolutional_stem`）**：
  - 由 3 个串联的 `MobileOneBlock` 组成：`$3\times 3$ Conv (stride=2)` $\rightarrow$ `$3\times 3$ DWConv (stride=2)` $\rightarrow$ `$1\times 1$ PWConv (stride=1)`，完成 **$4\times$ 空间下采样** 并将通道数升至 96。
  - **结构重参数化（Structural Reparameterization）**：训练时每个块包含并行的 $1\times 1$ 缩放分支与 BatchNorm 跳连分支以辅助优化；**推理时通过线性代数等价变换融合为单个无分支的标准卷积层**，降低硬件访存开销。

### 3.2 黄色块：`RepMixer Stage`（Stage 1, 2, 3 —— 源自 `FastViT` ICCV 2023）
- **针对的问题**：在浅层高分辨率特征图（$\frac{R}{4}, \frac{R}{8}, \frac{R}{16}$）上，硬件推理速度的主要瓶颈是 **显存访存开销（Memory Access Cost, MAC）** 而非算力（FLOPs）。常规 MetaFormer / ResNet 块中的残差连接 $Y = X + \text{Mixer}(\text{Norm}(X))$ 需要在内存中保留并反复读写完整的高分辨率张量 $X$。
- **具体实现（`mci.py` 中 `RepMixer` + `ConvFFN`）**：
  - **训练态**：`RepMixer` 计算 $Y = X + \alpha \cdot (\text{DWConv}_{3\times 3}(X) - \text{BN}(X))$，利用残差跳连与 BatchNorm 保证深层网络（Stage 1~3 共 `2 + 12 + 24 = 38` 层）稳定训练。
  - **推理态**：由于恒等映射 $X$、$\text{BN}(X)$ 和 $\text{DWConv}_{3\times 3}(X)$ 均为线性算子，导出推理模型时**直接合并为单一的 $3\times 3$ 深度可分离卷积 $Y = \text{DWConv}^*_{3\times 3}(X)$**，**完全消除了推理时的残差跳连（Skip Connection）和独立归一化层**。
  - **伴随的 `ConvFFN`**：每个 Token Mixer 后接一个 `ConvFFN`（前置一个 $7\times 7$ 深度卷积 + BN，再接扩展比为 $4.0$ 的两层 $1\times 1$ 逐点卷积与 GELU）。

### 3.3 蓝色块：`Patch Embed. Stride 2`（源自 `FastViT` + 本文附录 B 改造）
- **针对的问题**：网络总共进行 $64\times$ 空间下采样，若使用小卷积核（如 $2\times 2$ stride=2）下采样，感受野互不重叠，容易丢失细粒度空间信息。
- **具体实现（`mci.py` 中 `PatchEmbed` 与论文附录 B）**：
  1. **重叠大核深度卷积下采样**：采用 `ReparamLargeKernelConv`（训练时为 $7\times 7$ DWConv 与 $3\times 3$ DWConv 并联，推理时融合为单个 stride=2 的 $7\times 7$ 深度卷积）接 $1\times 1$ `MobileOneBlock`。步长为 2 的 $7\times 7$ 大核具有高度重叠的感受野，在空间尺寸减半的同时将通道数翻倍（$C \rightarrow 2C$），将局部空间结构编码进高维通道。
  2. **移除 Squeeze-and-Excitation (`SE`) 模块（本文附录 B 的针对性修改）**：前作 `MobileCLIP-MCi2` 在下采样块中使用了 `SE` 模块；但 `SE` 需要对整个特征图做全局平均池化同步，在高分辨率图像推理时显著增加端侧延迟，因此 `FastViTHD` 移除了各阶段间的 `SE` 模块。

### 3.4 绿色块：`Self Attention Stage`（Stage 4 & Stage 5 —— 本文 Sec. 3.2 核心改造）
- **针对的问题**：如论文 Fig. 3 所示，若按传统 4 阶段混合架构放大模型（`FastViT-Naive Scaling`，深度 `[2, 12, 16, 6]`，维度 `[128, 256, 512, 1024]`），高分辨率图像在第 3、4 阶段的特征图依然过大，导致自注意力耗时随分辨率急剧上升。
- **具体实现（`mci.py` 中 `fastvithd` 与 `AttentionBlock`）**：
  1. **新增第 5 阶段与第 4 个下采样层**：将各阶段深度设为 `[2, 12, 24, 4, 2]`，通道维度设为 `[96, 192, 384, 768, 1536]`，总下采样率达到 **$64\times$**。
  2. **推迟自注意力算子**：仅在 **Stage 4（$32\times$ 下采样，4 层，768 维）** 和 **Stage 5（$64\times$ 下采样，2 层，1536 维）** 使用多头自注意力 `MHSA`。
  3. **条件位置编码（`RepCPE`）**：在进入 Stage 4 和 Stage 5 之前，各插入一个基于 $7\times 7$ 深度卷积的可重参数化条件位置编码（`RepCPE`），使模型能够直接适应不同输入分辨率，无需像传统 ViT 那样对固定长度的绝对位置编码做插值。

### 3.5 紫色块：`(Learned) Pool` 与圆圈 `C`（Sec. 3.1.1 多尺度特征实验 vs. 实际开源实现）
- **论文设计（Sec. 3.1.1 & Table 2）**：为了补充浅层阶段的细粒度信息，作者在 `FastViT` 上实验了用 **2D 深度可分离卷积（DWConv）** 将早期阶段特征池化至末层空间尺寸，并沿通道拼接（`C`）。在 Table 2 中，该操作使 `FastViT` 的 Avg-5 从 `62.6%` 提升至 `62.9%`（+0.3%）。
- **开源代码实现对照（`mobileclip_encoder.py`）**：在官方发布的 `apple/ml-fastvlm` 代码库中，`FastViTHD` 并未启用多尺度分支拼接，而是直接提取 Stage 5 末尾特征图（`image_embeddings`，维度 $1536 \times \frac{R}{64} \times \frac{R}{64}$）展平后送入 MLP `Connector`。
