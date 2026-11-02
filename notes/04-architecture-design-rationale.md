# Figure 2 深度拆解：为什么 `FastViTHD` 要设计成这样的架构？

在阅读 FastVLM 原文时，一个非常普遍且敏锐的疑问是：**“为什么论文对 Figure 2 的架构设计过程着墨极少，几乎没解释为什么要这样搭积木，反而花了大量篇幅做实验评测？”**

原因有两个：
1. **研究脉络的继承性（苹果团队的“四部曲”演进）**：FastVLM 并不是从零发明了一个编码器，而是同一批 Apple 核心作者（Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Oncel Tuzel 等）在连续三篇前作——**`MobileOne` (CVPR 2023)** $\rightarrow$ **`FastViT` (ICCV 2023)** $\rightarrow$ **`MobileCLIP` (CVPR 2024)**——基础上的第 4 代演进。Figure 2 中几乎所有基础算子（`Convolutional Stem`、`Patch Embed`、`RepMixer`、`ConvFFN`）的设计哲学，都在前作中完成了推导与消融；而在 FastVLM 正文中，作者默认读者已知晓 `FastViT / MobileCLIP`，只用短短两段话（Sec. 3.2 与附录 B）交代了**为了适配高分辨率 VLM 而做的增量结构改造（Delta）**。
2. **Figure 2 其实是一张“缝合示意图”**：它把 **Sec. 3.1.1 在 `FastViT` 上尝试的多尺度池化拼接（`(Learned) Pool` + `C`）** 和 **Sec. 3.2 提出的 5 阶段 `FastViTHD` 主干** 画在了同一张图里（更有意思的是，对照 Apple 官方开源代码 `apple/ml-fastvlm` 可以发现，最终发布的 `FastViTHD` 甚至没有启用紫色的多尺度池化分支，而是直接取第 5 阶段输出）。

下面我们结合**前作论文的设计推导**与 **FastVLM 官方开源代码（`llava/model/multimodal_encoder/mobileclip/mci.py`）**，把 Figure 2 从左到右的每一个模块**“针对什么痛点、为什么这样设计”**彻底讲透。

---

## 1. 宏观结构：为什么要用「前卷积 + 后自注意力」的金字塔混合架构？

### 针对的特定问题
标准 ViT（如 ViT-L/14、SigLIP）是**各向同性（Isotropic）**架构：图像一进来就切成 $14\times 14$ 的 Patch，从第 1 层到第 24 层保持完全相同的空间分辨率（Token 数 $N = (R/14)^2$），且每一层都做全局多头自注意力（MHSA）。
- 当输入分辨率 $R$ 升至 $1024\times 1024$ 时，Token 数高达 $5350$ 个；
- 在网络的**浅层和中间层**，模型其实只需要提取边缘、笔画、局部纹理等低级视觉特征，此时做复杂度为 $O(N^2)$ 的全局自注意力完全是算力浪费，且会导致高分辨率下显存带宽与计算延迟爆炸。

### 设计思路（Hybrid Pyramid）
模仿生物视觉皮层与经典 CNN 金字塔（但保留 Transformer 的全局语义能力）：
- **浅层高分辨率阶段（Stage 1 ~ 3，黄色块）**：特征图空间尺寸大（$\frac{R}{4}, \frac{R}{8}, \frac{R}{16}$）、通道数窄（$96, 192, 384$）。只用 $O(N)$ 线性复杂度的**局部深度卷积算子（`RepMixer`）**快速聚合局部特征（如文档中的字符笔画、表格边线）。
- **深层低分辨率阶段（Stage 4 ~ 5，绿色块）**：特征图经过多次下采样后空间尺寸极小（$\frac{R}{32}, \frac{R}{64}$）、通道数极宽（$768, 1536$）。此时每个位置已经代表一个高级局部区域，再用 **全局多头自注意力（`MHSA`）** 进行跨区域的全局语义关联与上下文对齐，输出给 LLM。

---

## 2. 逐模块拆解 Figure 2：每个组件为什么这样设计？

![FastVLM Overview Architecture](../paper/figures/main_diagram.png)

### 模块 ①：橙色块 `Convolutional Stem`（源自 `MobileOne` & `FastViT`）
- **为什么不用 ViT 的 $14\times 14$ 非重叠线性切片（Patchify）？**
  - ViT 直接用 stride=14 的大步长卷积把图像切成互不重叠的方块，破坏了相邻像素间的局部连续性（尤其是小字号文字正好被切在边界上时），且无法提供平移不变性先验。
- **具体怎么设计的（见源码 `convolutional_stem`）？**
  - 使用 **3 个串联的 `MobileOneBlock`**（`$3\times 3$ Conv (stride=2)` $\rightarrow$ `$3\times 3$ DWConv (stride=2)` $\rightarrow$ `$1\times 1$ PWConv (stride=1)`），平滑地将输入图像下采样 **$4\times$**，并将通道数扩展至 $96$。
  - **训练态过参数化 vs. 推理态极简融合**：训练时每个块包含并行的卷积分支与恒等映射（Skip Connection）以保证梯度顺畅；**推理时通过结构重参数化（Structural Reparameterization）数学合并为 3 个没有任何分支的普通卷积层**，在硬件（如 Apple Neural Engine）上延迟极低。

---

### 模块 ②：黄色块 `RepMixer Stage`（Stage 1, 2, 3 —— 源自 `FastViT` ICCV 2023）
- **针对的特定问题：为什么在浅层不用普通 ResNet 残差块、ConvNeXt 块或 PoolFormer？**
  - 在现代端侧加速器（GPU / NPU / ANE）上，浅层高分辨率特征图（如 $256\times 256$）的瓶颈往往不是算力（FLOPs），而是**访存开销（Memory Access Cost, MAC）**！
  - 传统的残差连接 $Y = X + \text{Mixer}(\text{Norm}(X))$ 必须在芯片缓存/显存中把庞大的特征图 $X$ 先保存下来，等算完 $\text{Norm}$ 和 $\text{Mixer}$ 后再读出来做逐元素相加。这种跨分支的张量读写在高分辨率下非常耗时。
- **`RepMixer` 的核心巧思（推理时彻底消灭残差分支与归一化层！）**：
  - **训练阶段**：`RepMixer` 的公式为：
    $$Y = X + \alpha \cdot \left(\text{DWConv}_{3\times 3}(X) - \text{BN}(X)\right)$$
    既有残差连接又有 BatchNorm，保证 38 层深度的网络能够稳定收敛。
  - **推理阶段**：因为恒等映射 $X$、$\text{BN}(X)$ 和 $\text{DWConv}_{3\times 3}(X)$ 全部是对 $X$ 的线性算子，在导出推理模型时，它们可以**精确合并（融合）为唯一一个 $3\times 3$ 深度可分离卷积层**：
    $$Y = \text{DWConv}^*_{3\times 3}(X)$$
    **推理时没有任何残差跳连（Skip Connection），也没有独立的 Normalization 层**！这就是为什么 `FastViTHD` 可以在前 3 个高分辨率阶段堆叠多达 `2 + 12 + 24 = 38` 层却依然快得惊人。
- **配套的 `ConvFFN`**：在每个 `RepMixer` 后面接一个 `ConvFFN`（前置一个 $7\times 7$ 深度卷积来扩大感受野，再接扩展比为 $4.0$ 的两层 $1\times 1$ 逐点卷积 MLP）。

---

### 模块 ③：蓝色块 `Patch Embed. Stride 2`（源自 `FastViT` + 本文附录 B 的高分辨率改造）
- **针对的特定问题：连续 4 次 $2\times$ 下采样（共 $64\times$），如何防止细粒度信息（如小字）在下采样中丢失？**
  - 如果像普通 CNN 那样用 $2\times 2$（stride=2）的池化或小卷积做下采样，感受野互不重叠，高倍率下采样会导致严重的空间混叠（Aliasing）和细节丢失。
- **具体怎么设计的（见源码 `PatchEmbed` 与论文附录 B）？**
  - 每个蓝色 `Patch Embed` 层采用 **`ReparamLargeKernelConv`（训练时为 $7\times 7$ DWConv + $3\times 3$ DWConv 并行，推理时融合为单个 stride=2 的 $7\times 7$ 大核深度卷积）+ $1\times 1$ 逐点卷积**。
  - **为什么用 $7\times 7$ 大核做 stride=2 下采样？** 因为 $7\times 7$ 窗口在步长为 2 时有高度重叠的感受野，能够在空间尺寸减半的同时，把周围 $7\times 7$ 邻域内的空间结构信息（比如字符的相对笔画位置）平滑地编码进**翻倍的通道维度（$C \rightarrow 2C$）**中。
  - **本文（FastVLM 附录 B）专门针对高分辨率做的关键删减——去掉 Squeeze-and-Excitation (SE)**：
    - 在前作 `MobileCLIP-MCi2` 中，下采样层内部带有 `SE` 通道注意力模块。
    - 但在 FastVLM 中，作者实测发现：`SE` 模块需要对整张高分辨率特征图做全局平均池化（Global AvgPool）同步，这在高分辨率图像推理时会引发严重的硬件延迟卡顿！因此 **`FastViTHD` 果断移除了 Stage 间所有的 `SE` 模块**。

---

### 模块 ④：绿色块 `Self Attention Stage`（Stage 4 & Stage 5 —— **本文 Sec. 3.2 最核心的架构创新！**）

这是 **FastVLM 相比前作 `FastViT / MobileCLIP` 改动最大、也是整篇论文真正属于架构设计贡献的地方**：

- **特定问题：为什么不能直接把前作 4 阶段的 `FastViT` 加宽加深来用于高分辨率 VLM？**
  - 前作 `FastViT`（以及同类混合模型 `ViTamin`、`MOAT`）都是 **4 阶段设计**（Stem $4\times$ + 3 个 Patch Embed $2^3=8\times$，总下采样率 $16\times$ 或 $32\times$）。
  - 当作者尝试把 35.7M 的 `FastViT` 放大到 125M 参数（论文 Fig. 3 中的 `FastViT-Naive Scaling`，4 阶段深度 `[2, 12, 16, 6]`，维度 `[128, 256, 512, 1024]`）并输入 $768^2 \sim 1024^2$ 的高分辨率图像时，发现了两个致命问题：
    1. **编码器内部自注意力太慢（Fig. 3 红线）**：在 $16\times$ 下采样（Stage 3）和 $32\times$ 下采样（Stage 4）的特征图上堆叠大量自注意力层和 SE 层，当输入为 $1024\times 1024$ 时，特征图仍有 $64\times 64 = 4096$ 个位置，导致编码延迟飙升，甚至比纯卷积的 `ConvNeXt-L` 还要慢！
    2. **输出的视觉 Token 依然太多**：总下采样率 $32\times$ 意味着在 $1024\times 1024$ 下会输出 $(1024/32)^2 = 1024$ 个 Token，导致后面的 LLM Prefill 耗时依然偏高。
- **`FastViTHD` 的破局设计（引入第 5 阶段，把自注意力推迟到 $32\times$ 和 $64\times$ 下采样）**：
  1. **增加第 4 个 `Patch Embed. Stride 2` 和 `Stage 5`**：使总空间下采样率从 $32\times$ 翻倍至 **$64\times$**，通道维度继续翻倍至 **$1536$**。
  2. **重排算子分布（`[2, 12, 24, 4, 2]`）**：
     - 把 Stage 3 彻底改成纯卷积 `RepMixer`（24 层），让 $16\times$ 下采样率上完全不跑自注意力；
     - **Stage 4（$32\times$ 下采样，768 维）**只跑 **4 层 `MHSA`**；
     - **Stage 5（$64\times$ 下采样，1536 维）**跑 **2 层超宽通道的 `MHSA`**（进入 Stage 4 和 Stage 5 前各加一个可重参数化的 $7\times 7$ 条件位置编码 `RepCPE`，使模型天然适应任意动态分辨率而无需插值位置编码！）。
  3. **一石二鸟的效果**：
     - 视觉编码器内部的自注意力只在极小的张量（$1024^2$ 输入下仅 $32\times 32$ 和 $16\times 16$）上计算，**编码延迟直接降低数倍（Fig. 3 蓝线）**；
     - 输出给 LLM 的视觉 Token 数直接减少 **$4\times$**（从 1024 个降为 **256 个**），**大幅降低了 LLM 的 Prefill 延迟**；
     - 同时因为 Stage 5 的通道宽度高达 **1536 维**（ConvFFN 隐藏层高达 $1536 \times 4 = 6144$ 维），单个 Token 的语义承载力极强，保证了压缩 Token 数量的同时不损失 VLM 精度！

---

### 模块 ⑤：紫色块 `(Learned) Pool` 与圆圈 `C`（Sec. 3.1.1 的多尺度特征融合——以及代码中的“彩蛋”）

- **为什么 Figure 2 里画了从前面阶段引出的 3 个紫色 `(Learned) Pool` 分支和拼接圆圈 `C`？**
  - 这是作者在 **Sec. 3.1.1（Table 2）** 探索 `FastViT` 时提出的设计：由于层级化网络不断下采样，浅层阶段包含更精细的局部空间信息，深层阶段包含高层语义信息。作者尝试用 **2D 深度可分离卷积（DWConv）** 将浅层阶段特征池化到与最后一层相同的空间尺寸，再沿通道拼接（`C`）送入 `Connector`。
- **为什么你在读论文时会觉得这部分和后文割裂？（关键洞察）**
  - 看 **Table 2**：在 `FastViT` 上加入这个多尺度融合后，Avg-5 其实只从 `62.6%` 微弱提升到了 `62.9%`（仅 **+0.3%**）。
  - 更重要的是，如果我们直接查看 Apple 官方开源的 `ml-fastvlm` 代码库（`llava/model/multimodal_encoder/mobileclip_encoder.py` 第 60–68 行与 `mci.py` 第 1338–1361 行）：
    ```python
    def feature_select(self, image_forward_outs):
        # Features from penultimate layer
        image_features = image_forward_outs["image_embeddings"]
        B, C, H, W = image_features.shape
        image_features = image_features.reshape(B, C, H*W).transpose(1, 2)
        return image_features
    ```
    **最终开源并取得 SOTA 成绩的 `FastViTHD` 模型，实际上根本没有使用 Figure 2 右下角那三个紫色的 `(Learned) Pool` 分支**，而是直接把 `Stage 5` 输出的特征图（经 `conv_exp` 投影后）展平喂给了 `Connector`！
  - 也就是说，Figure 2 是作者为了在一张图里同时展示 **Sec. 3.1.1（多尺度池化探索）** 和 **Sec. 3.2（5 阶段 `FastViTHD` 设计）** 而绘制的综合示意图。

---

## 3. 总结：在 Presentation 中应该如何向听众讲解 Figure 2？

在课堂展示时，如果你能把这个**“跨越四篇论文的设计演进 + Figure 2 的真实设计逻辑”**讲出来，深度会远超普通照本宣科的汇报：

1. **先点明背景（为什么论文正文写得简略）**：
   > “大家看论文正文可能会发现，作者对 Figure 2 内部算子的解释非常简略。这是因为 `FastViTHD` 建立在 Apple 团队此前连续三篇底层架构工作（`MobileOne` $\rightarrow$ `FastViT` $\rightarrow$ `MobileCLIP`）之上。”
2. **再用三句话讲清 Figure 2 的三大核心设计逻辑**：
   - **第一，算子级减负（来自 `FastViT` 的 `RepMixer`）**：前 3 个阶段处理高分辨率特征图时，利用结构重参数化技术在推理阶段**消除了所有残差跳连（Skip Connections）和归一化层**，折叠为单一深度卷积，解决了高分辨率特征图的显存带宽（MAC）瓶颈。
   - **第二，下采样保真（来自 $7\times 7$ 大核 `Patch Embed` 与去 `SE` 化）**：每次 $2\times$ 下采样都使用 $7\times 7$ 重叠大核深度卷积并将通道数翻倍，把空间细节编码进通道维度；同时移除了高分辨率下拖慢硬件推理的 `SE` 模块。
   - **第三，阶段级重构（本文 `FastViTHD` 的核心创新——第 5 阶段与 $64\times$ 下采样）**：为了解决传统 4 阶段混合模型放大后在高分辨率下自注意力变慢的问题，新增了第 5 阶段，把昂贵的多头自注意力（`MHSA`）严格限制在已经下采样了 **$32\times$（Stage 4）和 $64\times$（Stage 5，1536 维超宽通道）**的浓缩特征图上，从而一举实现了**视觉编码极快**与**输出视觉 Token 数量减少 $16\times$** 的双重目标。
