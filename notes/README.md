# Reading Notes & Technical Background

本目录整理了 FastVLM 论文的精读总结、架构参数与数据表、结合前作与官方源码整理的 `FastViTHD` 架构设计逻辑拆解，以及文献讲解的核心架构主线。

```text
notes/
|-- README.md                               # 本目录索引
|-- 01-paper-summary.md                     # 论文精读总结（背景痛点、FastViTHD 架构、帕累托分析与实验结果）
|-- 02-architecture-and-tables.md           # 核心架构逐层参数、Token 缩放公式与实验数据表
|-- 03-architecture-design-rationale.md     # Figure 2 与 FastViTHD 架构设计逻辑拆解（结合前作与源码）
`-- 04-storyline.md                         # 文献讲解核心主线（6~7 分钟四层递进架构剖析与演进脉络）
```

## 笔记导航

| 文档 | 核心内容 |
|---|---|
| [01-paper-summary.md](01-paper-summary.md) | FastVLM 论文全文深度精读：研究背景与高分辨率双重瓶颈、`FastViTHD` 5 阶段混合架构设计、帕累托最优分析与实验结论。 |
| [02-architecture-and-tables.md](02-architecture-and-tables.md) | 架构参数与实验数据速查手册：`FastViTHD` 逐阶段参数表、不同视觉编码器的 Token 数计算公式与 Table 1～6 关键数据对照。 |
| [03-architecture-design-rationale.md](03-architecture-design-rationale.md) | 架构演进与设计逻辑拆解：结合前作 `FastViT`、`MobileCLIP` 与官方源码（`mci.py`），剖析 `RepMixer` 重参数化、大核卷积、5 阶段 $64\times$ 下采样及训练蒸馏方案。 |
| [04-storyline.md](04-storyline.md) | 6～7 分钟文献讲解核心主线：从传统 ViT 痛点到算子级、阶段级与系统级的四层递进技术剖析。 |
