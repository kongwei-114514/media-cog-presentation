# Paper & Figures

本目录存放 **FastVLM** 及其前序系列工作（**FastViT**、**MobileCLIP**）的论文原文 PDF，以及从各篇 arXiv 官方 LaTeX 源码包中提取转换的 300 DPI 高清图表。

```text
paper/
|-- README.md                               # 本目录索引
|-- FastVLM_2412.13303.pdf                  # FastVLM 论文原文 (CVPR 2025, arXiv:2412.13303v2)
|-- FastViT_2303.14189.pdf                  # 前作 FastViT 论文原文 (ICCV 2023, arXiv:2303.14189)
|-- MobileCLIP_2311.17049.pdf               # 前作 MobileCLIP 论文原文 (CVPR 2024, arXiv:2311.17049)
`-- figures/                                # 从 LaTeX 源码矢量图转换的 300 DPI PNG 图像
    |-- acc_vs_latency_qwen.png             # FastVLM Fig. 1(a): Qwen2-0.5B 下 Avg-5 精度 vs. TTFT
    |-- acc_vs_latency_vicuna.png           # FastVLM Fig. 1(b): Vicuna-7B 下 Avg-5 精度 vs. TTFT
    |-- main_diagram.png                    # FastVLM Fig. 2: FastVLM 与 FastViTHD 5 阶段整体架构图
    |-- resolution_scaling_trend.png        # FastVLM Fig. 3: FastViTHD vs. FastViT-Naive 与 ConvNeXt-L 延迟曲线
    |-- decoder_encoder_ablation.png        # FastVLM Fig. 4: 跨分辨率与 LLM 尺寸的帕累托最优曲线
    |-- ttft_breakdown.png                  # FastVLM Fig. 5: 不同分辨率下视觉编码与 LLM Prefill 延迟占比
    |-- static_vs_dynamic_res.png           # FastVLM Fig. 6: 静态整图缩放 (Static) vs. 动态分块 (AnyRes)
    |-- fastvit_arch.png                    # FastViT Fig. 2: Stem / PatchEmbed / RepMixer / ConvFFN 训练与推理重参数化微观结构图
    |-- fastvit_reparam_latency.png         # FastViT Fig. 3: RepMixer 消除残差跳连在 224~1024 分辨率下的延迟收益对比
    `-- mobileclip_dr_schematic.png         # MobileCLIP Fig. 2: DataCompDR 多模态强化预训练（合成描述+教师集成蒸馏）示意图
```

## 图表速览与对应论文出处

| 来源论文与图号 | 文件 | 核心内容 |
|---|---|---|
| **FastVLM Fig. 1(a)** | [acc_vs_latency_qwen.png](figures/acc_vs_latency_qwen.png) | 搭配 Qwen2-0.5B 时，`FastViTHD` 比主流 CLIP 编码器实现 $3.2\times$ 更快 TTFT。 |
| **FastVLM Fig. 1(b)** | [acc_vs_latency_vicuna.png](figures/acc_vs_latency_vicuna.png) | 搭配 Vicuna-7B 时，`FastViTHD` 比主流视觉编码器实现 $>2.3\times$ 更快 TTFT。 |
| **FastVLM Fig. 2** | [main_diagram.png](figures/main_diagram.png) | `FastViTHD` 5 阶段（3 阶段 `RepMixer` + 2 阶段 `MHSA`）与 $64\times$ 下采样宏观架构图。 |
| **FastVLM Fig. 3** | [resolution_scaling_trend.png](figures/resolution_scaling_trend.png) | 5 阶段 `FastViTHD` 相比 4 阶段朴素放大 `FastViT-Naive` 与 `ConvNeXt-L` 的编码延迟对比。 |
| **FastVLM Fig. 4** | [decoder_encoder_ablation.png](figures/decoder_encoder_ablation.png) | `FastViTHD` vs. `FastViT` 在不同 `(Resolution, LLM)` 组合下的帕累托最优曲线。 |
| **FastVLM Fig. 5** | [ttft_breakdown.png](figures/ttft_breakdown.png) | 输入分辨率从 $256^2$ 增至 $1536^2$ 时，视觉编码延迟与 LLM Prefill 延迟的占比演进。 |
| **FastVLM Fig. 6** | [static_vs_dynamic_res.png](figures/static_vs_dynamic_res.png) | 静态直接缩放分辨率 vs. AnyRes 动态分块切图的精度-延迟对比。 |
| **FastViT Fig. 2** | [fastvit_arch.png](figures/fastvit_arch.png) | `Convolutional Stem`、`Patch Embed`、`RepMixer`、`ConvFFN`、`Attention + CPE` 的训练态/推理态重参数化微观结构图。 |
| **FastViT Fig. 3** | [fastvit_reparam_latency.png](figures/fastvit_reparam_latency.png) | `RepMixer` 消除残差跳连相比 `Pooling` 在 $224\times 224 \sim 1024\times 1024$ 分辨率下的真机延迟下降曲线（$1024^2$ 下降低 43.9%）。 |
| **MobileCLIP Fig. 2** | [mobileclip_dr_schematic.png](figures/mobileclip_dr_schematic.png) | `DataCompDR` 数据集强化流程（图像增强、合成描述生成与多教师集成特征蒸馏）。 |
