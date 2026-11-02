# Paper & Figures

本目录存放 FastVLM 论文原文 PDF 以及从 arXiv 官方 LaTeX 源码包中提取转换的 300 DPI 高清图表，方便直接阅读或在制作展示幻灯片时引用。

```text
paper/
|-- README.md                             # 本目录索引
|-- FastVLM_2412.13303.pdf                # FastVLM 论文原文 (arXiv:2412.13303v2, 含正文与附录共 20 页)
`-- figures/                              # 论文核心矢量图转换的 300 DPI PNG 图像
    |-- acc_vs_latency_qwen.png           # Fig. 1(a): Qwen2-0.5B 下不同视觉编码器的 Avg-5 精度 vs. TTFT
    |-- acc_vs_latency_vicuna.png         # Fig. 1(b): Vicuna-7B 下不同视觉编码器的 Avg-5 精度 vs. TTFT
    |-- main_diagram.png                  # Fig. 2: FastVLM 与 FastViTHD 5 阶段层级化混合架构全图
    |-- resolution_scaling_trend.png      # Fig. 3: FastViTHD vs. FastViT-Naive 与 ConvNeXt-L 的编码延迟曲线
    |-- decoder_encoder_ablation.png      # Fig. 4: FastViTHD vs. FastViT 跨分辨率与 LLM 尺寸的帕累托最优曲线
    |-- ttft_breakdown.png                # Fig. 5: 不同分辨率下视觉编码延迟与 LLM Prefill 延迟占比拆解
    `-- static_vs_dynamic_res.png         # Fig. 6: 静态整图缩放 (Static) vs. 动态分块 (AnyRes) 的精度-延迟对比
```

## 图表速览与对应论文小节

| 图号 | 文件 | 对应章节 | 核心结论 |
|---|---|---|---|
| **Fig. 1(a)** | [acc_vs_latency_qwen.png](figures/acc_vs_latency_qwen.png) | Sec. 1 | 搭配 Qwen2-0.5B 时，`FastViTHD` 比主流 CLIP 编码器实现 **$3.2\times$ 更快 TTFT**。 |
| **Fig. 1(b)** | [acc_vs_latency_vicuna.png](figures/acc_vs_latency_vicuna.png) | Sec. 1 | 搭配 Vicuna-7B 时，`FastViTHD` 比主流视觉编码器实现 **$>2.3\times$ 更快 TTFT**。 |
| **Fig. 2** | [main_diagram.png](figures/main_diagram.png) | Sec. 3.1–3.2 | `FastViTHD` 采用 5 阶段（3 阶段 `RepMixer` + 2 阶段 `MHSA`）与 $64\times$ 下采样及多尺度池化拼接。 |
| **Fig. 3** | [resolution_scaling_trend.png](figures/resolution_scaling_trend.png) | Sec. 3.2 | 在第 4、5 阶段（$32\times / 64\times$ 下采样）做自注意力显著降低了高分辨率编码延迟。 |
| **Fig. 4** | [decoder_encoder_ablation.png](figures/decoder_encoder_ablation.png) | Sec. 3.2.1 | `FastViTHD` 在所有 `(Resolution, LLM)` 组合上将帕累托前沿推高 $>2.5$ 分（或同精度快 $3\times$）。 |
| **Fig. 5** | [ttft_breakdown.png](figures/ttft_breakdown.png) | Sec. 3.2.1 | 当输入分辨率 $\ge 768\times 768$ 时，视觉编码器延迟反超 LLM Prefill 成为 TTFT 的主导部分。 |
| **Fig. 6** | [static_vs_dynamic_res.png](figures/static_vs_dynamic_res.png) | Sec. 3.2.2 | 静态直接缩放分辨率优于多网格切图（AnyRes），仅在 $1536^2$ 以上极限分辨率配 $2\times 2$ 切图有益。 |
