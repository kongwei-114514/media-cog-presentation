# Media & Cognition Presentation — FastVLM

清华大学「媒体与认知」课程文献展示仓库，聚焦于 Apple 发表在 CVPR 2025 的高效视觉语言模型论文 **FastVLM: Efficient Vision Encoding for Vision Language Models**（[arXiv:2412.13303](https://arxiv.org/abs/2412.13303)）。

## Repository layout

```text
media-cog-presentation/
|-- README.md                             # 仓库总览、论文信息与导航索引
|-- paper/                                # 论文原文 PDF 与提取的高清核心图表
|   |-- README.md                         # 论文与图表目录索引
|   |-- FastVLM_2412.13303.pdf            # FastVLM 论文原文 (arXiv:2412.13303v2)
|   `-- figures/                          # 从 arXiv LaTeX 源码提取的 300 DPI 核心配图
|       |-- acc_vs_latency_qwen.png       # Fig. 1(a): Qwen2-0.5B 下 Avg-5 精度 vs. TTFT
|       |-- acc_vs_latency_vicuna.png     # Fig. 1(b): Vicuna-7B 下 Avg-5 精度 vs. TTFT
|       |-- main_diagram.png              # Fig. 2: FastVLM 与 FastViTHD 5 阶段整体架构图
|       |-- resolution_scaling_trend.png  # Fig. 3: FastViTHD 与朴素放大的分辨率-延迟曲线
|       |-- decoder_encoder_ablation.png  # Fig. 4: 不同分辨率与 LLM 规模下的帕累托最优曲线
|       |-- ttft_breakdown.png            # Fig. 5: 不同分辨率下视觉编码与 LLM Prefill 延迟占比
|       `-- static_vs_dynamic_res.png     # Fig. 6: 静态分辨率缩放 vs. AnyRes 动态切图对比
|-- notes/                                # 文献深度总结、数据速查与展示讲解指南
|   |-- README.md                         # 笔记目录索引
|   |-- 01-paper-summary.md               # 论文深度精读总结（背景、动机、FastViTHD 架构与实验全解）
|   |-- 02-architecture-and-tables.md     # 核心架构参数、Token 计算公式与关键实验数据速查表
|   |-- 03-presentation-guide.md          # 课程展示讲解指南（叙事主线、分页蓝图、逐页讲稿与 Q&A 预案）
|   `-- 04-architecture-design-rationale.md # Figure 2 深度拆解：四代架构演进与各模块底层设计逻辑
`-- slides/                               # 幻灯片与讲稿素材目录
    `-- README.md                         # 幻灯片目录说明
```

## Quick navigation

| 用途 | 入口文档 |
|---|---|
| 阅读论文原文 PDF | [FastVLM_2412.13303.pdf](paper/FastVLM_2412.13303.pdf) |
| 浏览提取的论文高清原图（可直接用于 PPT） | [论文与图表索引](paper/README.md) |
| 掌握论文核心脉络、动机、架构设计与实验结论 | [01-paper-summary.md](notes/01-paper-summary.md) |
| 深入理解 Figure 2 各模块为什么这样设计（结合前作与开源代码） | [04-architecture-design-rationale.md](notes/04-architecture-design-rationale.md) |
| 查阅 `FastViTHD` 各阶段参数、Token 数量与端侧延迟对比表 | [02-architecture-and-tables.md](notes/02-architecture-and-tables.md) |
| 准备课程展示叙事逻辑、分页结构、逐页讲稿与问答（Q&A） | [03-presentation-guide.md](notes/03-presentation-guide.md) |

## Paper overview

- **Title**: *FastVLM: Efficient Vision Encoding for Vision Language Models*
- **Authors**: Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari (Apple)
- **Venue / arXiv**: CVPR 2025 / [arXiv:2412.13303v2](https://arxiv.org/abs/2412.13303)
- **Official Code**: [apple/ml-fastvlm](https://github.com/apple/ml-fastvlm)
- **Core Highlights**:
  - **混合视觉编码器 `FastViTHD`**：采用「前 3 阶段 `RepMixer` 卷积 + 后 2 阶段 `MHSA` 自注意力」的 5 阶段设计，总空间下采样率达 **$64\times$**，结合多尺度深度卷积池化特征融合，在 $1024\times 1024$ 高分辨率下仅输出 **256 个视觉 Token**（比 ViT-L/14 少 $16\sim 20\times$，比 FastViT 少 $4\times$）。
  - **极简帕累托最优**：无需任何后验 Token 剪枝模块或复杂分块（Tiling），仅通过原生缩放输入图像分辨率即可实现最优的「分辨率–延迟–精度」折中。
  - **端侧实测大幅加速**：在 M1 Max MacBook Pro（ANE + GPU）上实测，LLaVA-1.5 设定下实现 **$3.2\times$ 首字延迟（TTFT）加速**；相比最高分辨率（$1152\times 1152$）下的 LLaVA-OneVision-0.5B 实现 **$85\times$ TTFT 加速**（166 ms vs. 14,124 ms），且视觉编码器参数量小 **$3.4\times$**（125M vs. 430M）。
