# Media & Cognition Presentation — FastVLM

「媒体与认知」课程文献展示仓库，整理了 **FastVLM: Efficient Vision Encoding for Vision Language Models**（[arXiv:2412.13303](https://arxiv.org/abs/2412.13303), CVPR 2025）及其前序架构工作（**FastViT** [arXiv:2303.14189](https://arxiv.org/abs/2303.14189), ICCV 2023；**MobileCLIP** [arXiv:2311.17049](https://arxiv.org/abs/2311.17049), CVPR 2024）的论文原文、提取图表、架构技术精读笔记与展示幻灯片预览稿。

## Repository layout

```text
media-cog-presentation/
|-- README.md                                   # 仓库总览与导航索引
|-- paper/                                      # 论文原文 PDF 与提取的高清核心图表
|   |-- README.md                               # 论文与图表目录索引
|   |-- FastVLM_2412.13303.pdf                  # FastVLM 论文原文 (arXiv:2412.13303v2, CVPR 2025)
|   |-- FastViT_2303.14189.pdf                  # 前作 FastViT 论文原文 (arXiv:2303.14189, ICCV 2023)
|   |-- MobileCLIP_2311.17049.pdf               # 前作 MobileCLIP 论文原文 (arXiv:2311.17049, CVPR 2024)
|   `-- figures/                                # 从 arXiv LaTeX 源码提取的 300 DPI 核心配图
|       |-- acc_vs_latency_qwen.png             # FastVLM Fig. 1(a): Qwen2-0.5B 下 Avg-5 精度 vs. TTFT
|       |-- acc_vs_latency_vicuna.png           # FastVLM Fig. 1(b): Vicuna-7B 下 Avg-5 精度 vs. TTFT
|       |-- main_diagram.png                    # FastVLM Fig. 2: FastVLM 与 FastViTHD 5 阶段整体架构图
|       |-- resolution_scaling_trend.png        # FastVLM Fig. 3: FastViTHD 与朴素放大的分辨率-延迟曲线
|       |-- decoder_encoder_ablation.png        # FastVLM Fig. 4: 不同分辨率与 LLM 规模下的帕累托最优曲线
|       |-- ttft_breakdown.png                  # FastVLM Fig. 5: 不同分辨率下视觉编码与 LLM Prefill 延迟占比
|       |-- static_vs_dynamic_res.png           # FastVLM Fig. 6: 静态分辨率缩放 vs. AnyRes 动态切图对比
|       |-- fastvit_arch.png                    # FastViT Fig. 2: Stem / PatchEmb / RepMixer / ConvFFN 微观重参数化架构图
|       |-- fastvit_reparam_latency.png         # FastViT Fig. 3: RepMixer 消除残差跳连在 224~1024 分辨率下的延迟收益
|       `-- mobileclip_dr_schematic.png         # MobileCLIP Fig. 2: DataCompDR 多模态强化蒸馏预训练流程图
|-- notes/                                      # 文献精读总结与底层架构背景资料
|   |-- README.md                               # 笔记目录索引
|   |-- 01-paper-summary.md                     # FastVLM 论文精读总结（背景痛点、FastViTHD 架构与实验分析）
|   |-- 02-architecture-and-tables.md           # 核心架构参数、Token 计算公式与实验数据对照表
|   |-- 03-architecture-design-rationale.md     # 架构演进与设计逻辑拆解（结合 FastViT、MobileCLIP 与官方源码）
|   `-- 04-storyline.md                         # 文献讲解核心主线（6~7 分钟四层递进架构剖析与演进脉络）
`-- draft/                                      # 课程展示大纲、逐页设计与 Marp 幻灯片预览稿
    |-- README.md                               # 展示草稿与配图目录索引
    |-- 01-pre-outline.md                       # 7 分钟展示定稿大纲（五部分逻辑链与核心用图规划）
    |-- 02-slides.md                            # 幻灯片逐页设计规范、页面约束与配图规划
    |-- 03-slides-preview.md                    # 17 页 Marp 幻灯片预览源码（含逐页讲稿备注）
    |-- 03-slides-preview.pdf                   # 导出的 17 页 Marp 幻灯片预览 PDF
    `-- figures/                                # 幻灯片专用局部放大、红框标注图与自绘示意图
        |-- fastvit_stage_column_annotated_p7.png   # FastViT 4 阶段竖版标注图（与 FastViTHD 等高并排对比用）
        |-- fastvithd_multiscale_branches.png       # FastViTHD 多尺度特征分支论文原图局部裁剪
        |-- fastvithd_stage_column_annotated.png    # FastViTHD 5 阶段竖版标注图（标出 4x~64x 下采样与自注意力）
        |-- fastvithd_stages_p9_annotated.png       # FastViTHD 横版三点核心设计标注图（①先看细节 ②通道数 ③多尺度）
        |-- p2_token_count_chart.png                # ViT 在 336² vs. 1024² 分辨率下的 Token 数平方增长对比图
        |-- p4_pipeline.png                         # VLM 三框流水线示意图（高亮视觉编码器并嵌入 FastViT 结构）
        |-- p11_token_grid_coarse.png               # 下采样网格变粗导致细小文字丢失的直观示意图
        |-- p13_channel_stack.png                   # “空间换通道”特征图尺寸与通道维度的直观解释图
        |-- p14_multiscale_concat.png               # 多尺度特征经 Learned Pool 对齐并在通道维拼接（C）示意图
        |-- p15_same_tokens_resolution.png          # 相同 256 个 Token 下 ViT (224²) vs. FastViTHD (1024²) 清晰度对比
        |-- zoomA_fastvit_stage_column.png          # FastViT 左侧 4 阶段金字塔主干放大图（无额外标注）
        |-- zoomA_fastvit_stage_column_annotated.png# FastViT 左侧 4 阶段主干放大标注图（标出 4x~32x 与 1024 Token）
        |-- zoomB_fastvit_stage4_attention.png      # FastViT Stage 4 自注意力模块局部放大图
        |-- zoomC_fastvit_repmixer.png              # FastViT RepMixer 训练态多分支与推理态单卷积合成放大图
        `-- src/                                    # 自绘示意图的 HTML 源文件与素材
```

## Quick navigation

| 类别 | 内容 | 入口文档 |
|---|---|---|
| **展示幻灯片 (Draft)** | 展示草稿与配图索引 | [draft/README.md](draft/README.md) |
| **展示幻灯片 (Draft)** | 7 分钟展示定稿大纲（五部分逻辑链） | [01-pre-outline.md](draft/01-pre-outline.md) |
| **展示幻灯片 (Draft)** | 幻灯片逐页设计规范与配图方案 | [02-slides.md](draft/02-slides.md) |
| **展示幻灯片 (Draft)** | 17 页 Marp 幻灯片源码（含讲稿备注） | [03-slides-preview.md](draft/03-slides-preview.md) |
| **展示幻灯片 (Draft)** | 17 页幻灯片导出预览 PDF | [03-slides-preview.pdf](draft/03-slides-preview.pdf) |
| **精读笔记 (Notes)** | 笔记目录索引 | [notes/README.md](notes/README.md) |
| **精读笔记 (Notes)** | FastVLM 论文全文总结 | [01-paper-summary.md](notes/01-paper-summary.md) |
| **精读笔记 (Notes)** | 架构参数与实验数据速查表 | [02-architecture-and-tables.md](notes/02-architecture-and-tables.md) |
| **精读笔记 (Notes)** | 架构演进与设计原理拆解（`FastViT` $\rightarrow$ `MobileCLIP` $\rightarrow$ `FastViTHD`） | [03-architecture-design-rationale.md](notes/03-architecture-design-rationale.md) |
| **精读笔记 (Notes)** | 文献讲解核心主线（四层递进架构剖析） | [04-storyline.md](notes/04-storyline.md) |
| **论文与原图 (Paper)** | 论文提取图表索引 | [paper/README.md](paper/README.md) |
| **论文与原图 (Paper)** | FastVLM 论文原文 PDF (CVPR 2025) | [FastVLM_2412.13303.pdf](paper/FastVLM_2412.13303.pdf) |
| **论文与原图 (Paper)** | 前作 FastViT 论文原文 PDF (ICCV 2023) | [FastViT_2303.14189.pdf](paper/FastViT_2303.14189.pdf) |
| **论文与原图 (Paper)** | 前作 MobileCLIP 论文原文 PDF (CVPR 2024) | [MobileCLIP_2311.17049.pdf](paper/MobileCLIP_2311.17049.pdf) |
