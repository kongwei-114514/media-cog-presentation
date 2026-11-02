# Papers & Figures

本目录存放 **FastVLM** 及其两篇直接前序工作（**FastViT**、**MobileCLIP**）的论文原文 PDF，以及从三篇论文 arXiv 官方 LaTeX 源码包中提取转换的 300 DPI 高清图表。

```text
paper/
|-- README.md                               # 本目录索引
|-- FastVLM_2412.13303.pdf                  # FastVLM 论文原文 (arXiv:2412.13303v2, CVPR 2025)
|-- FastViT_2303.14189.pdf                  # 前作 FastViT 论文原文 (arXiv:2303.14189, ICCV 2023)
|-- MobileCLIP_2311.17049.pdf               # 前作 MobileCLIP 论文原文 (arXiv:2311.17049, CVPR 2024)
`-- figures/                                # 论文核心矢量图转换的 300 DPI PNG 图像
    |-- acc_vs_latency_qwen.png             # FastVLM Fig. 1(a): Qwen2-0.5B 下不同视觉编码器的 Avg-5 精度 vs. TTFT
    |-- acc_vs_latency_vicuna.png           # FastVLM Fig. 1(b): Vicuna-7B 下不同视觉编码器的 Avg-5 精度 vs. TTFT
    |-- main_diagram.png                    # FastVLM Fig. 2: FastVLM 与 FastViTHD 5 阶段层级化混合架构全图
    |-- resolution_scaling_trend.png        # FastVLM Fig. 3: FastViTHD vs. FastViT-Naive 与 ConvNeXt-L 的编码延迟曲线
    |-- decoder_encoder_ablation.png        # FastVLM Fig. 4: FastViTHD vs. FastViT 跨分辨率与 LLM 尺寸的帕累托最优曲线
    |-- ttft_breakdown.png                  # FastVLM Fig. 5: 不同分辨率下视觉编码延迟与 LLM Prefill 延迟占比拆解
    |-- static_vs_dynamic_res.png           # FastVLM Fig. 6: 静态整图缩放 (Static) vs. 动态分块 (AnyRes) 的精度-延迟对比
    |-- fastvit_arch.png                    # FastViT Fig. 2: Stem / PatchEmb / RepMixer / ConvFFN 训练态与推理态重参数化微观结构图
    |-- fastvit_reparam_latency.png         # FastViT Fig. 3: RepMixer 消除残差跳连在 224~1024 分辨率下的真机延迟下降曲线 (最高降 43.9%)
    `-- mobileclip_dr_schematic.png         # MobileCLIP Fig. 2: DataCompDR 多模态合成描述与教师集成蒸馏流程图
```

## 图表速览与对应论文出处

| 出处与图号 | 文件 | 核心内容 |
|---|---|---|
| **FastVLM Fig. 1(a)** | [acc_vs_latency_qwen.png](figures/acc_vs_latency_qwen.png) | 搭配 Qwen2-0.5B 时，`FastViTHD` 比主流 CLIP 编码器实现 **$3.2\times$ 更快 TTFT**。 |
| **FastVLM Fig. 1(b)** | [acc_vs_latency_vicuna.png](figures/acc_vs_latency_vicuna.png) | 搭配 Vicuna-7B 时，`FastViTHD` 比主流视觉编码器实现 **$>2.3\times$ 更快 TTFT**。 |
| **FastVLM Fig. 2** | [main_diagram.png](figures/main_diagram.png) | `FastViTHD` 的 5 阶段（3 阶段 `RepMixer` + 2 阶段 `MHSA`）与 $64\times$ 下采样宏观框图。 |
| **FastVLM Fig. 3** | [resolution_scaling_trend.png](figures/resolution_scaling_trend.png) | 在第 4、5 阶段（$32\times / 64\times$ 下采样）做自注意力显著降低了高分辨率编码延迟。 |
| **FastVLM Fig. 4** | [decoder_encoder_ablation.png](figures/decoder_encoder_ablation.png) | `FastViTHD` 在所有 `(Resolution, LLM)` 组合上将帕累托前沿推高 $>2.5$ 分（或同精度快 $3\times$）。 |
| **FastVLM Fig. 5** | [ttft_breakdown.png](figures/ttft_breakdown.png) | 当输入分辨率 $\ge 768\times 768$ 时，视觉编码器延迟反超 LLM Prefill 成为 TTFT 的主导部分。 |
| **FastVLM Fig. 6** | [static_vs_dynamic_res.png](figures/static_vs_dynamic_res.png) | 静态直接缩放分辨率优于多网格切图（AnyRes），仅在 $1536^2$ 以上极限分辨率配 $2\times 2$ 切图有益。 |
| **FastViT Fig. 2** | [fastvit_arch.png](figures/fastvit_arch.png) | 详细展示 `Stem`、`Patch Emb`、`RepMixer`、`ConvFFN`、`CPE` 从训练态多分支折叠为推理态单卷积的过程。 |
| **FastViT Fig. 3** | [fastvit_reparam_latency.png](figures/fastvit_reparam_latency.png) | 相同 1.8G FLOPs 下，`RepMixer` 消除残差跳连访存（MAC）在 $1024\times 1024$ 分辨率下降低 43.9% 延迟。 |
| **MobileCLIP Fig. 2** | [mobileclip_dr_schematic.png](figures/mobileclip_dr_schematic.png) | `DataCompDR` 数据集强化流程：合成密集描述 + 图像增广参数 + 教师模型集成特征蒸馏。 |
