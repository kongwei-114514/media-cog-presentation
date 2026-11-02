# Media & Cognition Presentation — FastVLM

「媒体与认知」课程文献仓库，整理了 **FastVLM: Efficient Vision Encoding for Vision Language Models**（[arXiv:2412.13303](https://arxiv.org/abs/2412.13303), CVPR 2025）及其前序工作（**FastViT** ICCV 2023、**MobileCLIP** CVPR 2024）的论文原文、提取图表与架构技术总结。

## Repository layout

```text
media-cog-presentation/
|-- README.md                               # 仓库总览与导航索引
|-- paper/                                  # 论文原文 PDF 与提取的高清核心图表
|   |-- README.md                           # 论文与图表目录索引
|   |-- FastVLM_2412.13303.pdf              # FastVLM 论文原文 (CVPR 2025, arXiv:2412.13303v2)
|   |-- FastViT_2303.14189.pdf              # 前作 FastViT 论文原文 (ICCV 2023, arXiv:2303.14189)
|   |-- MobileCLIP_2311.17049.pdf           # 前作 MobileCLIP 论文原文 (CVPR 2024, arXiv:2311.17049)
|   `-- figures/                            # 从 arXiv LaTeX 源码提取的 300 DPI 核心配图
|       |-- acc_vs_latency_qwen.png         # FastVLM Fig. 1(a): Qwen2-0.5B 下 Avg-5 精度 vs. TTFT
|       |-- acc_vs_latency_vicuna.png       # FastVLM Fig. 1(b): Vicuna-7B 下 Avg-5 精度 vs. TTFT
|       |-- main_diagram.png                # FastVLM Fig. 2: FastVLM 与 FastViTHD 5 阶段整体架构图
|       |-- resolution_scaling_trend.png    # FastVLM Fig. 3: FastViTHD 与朴素放大的分辨率-延迟曲线
|       |-- decoder_encoder_ablation.png    # FastVLM Fig. 4: 不同分辨率与 LLM 规模下的帕累托最优曲线
|       |-- ttft_breakdown.png              # FastVLM Fig. 5: 不同分辨率下视觉编码与 LLM Prefill 延迟占比
|       |-- static_vs_dynamic_res.png       # FastVLM Fig. 6: 静态分辨率缩放 vs. AnyRes 动态切图对比
|       |-- fastvit_arch.png                # FastViT Fig. 2: 各模块训练态与推理态结构重参数化微观架构图
|       |-- fastvit_reparam_latency.png     # FastViT Fig. 3: RepMixer 消除残差跳连在不同分辨率下的延迟收益
|       `-- mobileclip_dr_schematic.png     # MobileCLIP Fig. 2: DataCompDR 多模态强化预训练示意图
`-- notes/                                  # 文献精读总结与底层架构背景资料
    |-- README.md                           # 笔记目录索引
    |-- 01-paper-summary.md                 # FastVLM 论文精读总结（背景痛点、FastViTHD 架构与实验分析）
    |-- 02-architecture-and-tables.md       # 核心架构参数、Token 计算公式与实验数据对照表
    `-- 03-architecture-design-rationale.md # FastViTHD 架构演进与设计逻辑拆解（结合 FastViT、MobileCLIP 与源码）
```

## Quick navigation

| 内容 | 入口文档 |
|---|---|
| FastVLM / FastViT / MobileCLIP 论文原文与图表索引 | [paper/README.md](paper/README.md) |
| FastVLM 论文全文总结 | [01-paper-summary.md](notes/01-paper-summary.md) |
| 架构参数与数据表 | [02-architecture-and-tables.md](notes/02-architecture-and-tables.md) |
| 四代架构演进与底层算子设计原理拆解（含 `FastViT` 微观结构图与消融） | [03-architecture-design-rationale.md](notes/03-architecture-design-rationale.md) |
