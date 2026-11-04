# Presentation Draft & Slide Figures

本目录存放 FastVLM 7 分钟课程展示的定稿大纲、逐页幻灯片设计规范、Marp 幻灯片预览稿（含逐页讲稿与导出的 PDF），以及专为幻灯片制作的论文局部放大标注图与自绘机制示意图。

```text
draft/
|-- README.md                                       # 本目录索引
|-- 01-pre-outline.md                               # 7 分钟展示定稿大纲（五部分逻辑链与核心用图规划）
|-- 02-slides.md                                    # 幻灯片逐页设计规范、全局约束与配图清单
|-- 03-slides-preview.md                            # 17 页 Marp 幻灯片预览源码（含每页讲稿注释）
|-- 03-slides-preview.pdf                           # 导出的 17 页 Marp 幻灯片预览 PDF
`-- figures/                                        # 幻灯片专用局部放大、红框标注图与自绘示意图
    |-- fastvit_stage_column_annotated_p7.png       # 预览稿 P9: FastViT 4 阶段竖版标注图（去插图，与 FastViTHD 等高并排）
    |-- fastvithd_multiscale_branches.png           # 预览稿 P13 备选: FastViTHD 多尺度特征分支原图局部裁剪
    |-- fastvithd_stage_column_annotated.png        # 预览稿 P9: FastViTHD 5 阶段竖版标注图（标出 4x~64x 下采样与自注意力）
    |-- fastvithd_stages_p9_annotated.png           # 预览稿 P11: FastViTHD 横版三点核心设计标注图（①先看细节 ②通道数 ③多尺度）
    |-- p2_token_count_chart.png                    # 预览稿 P2: ViT 在 336² vs. 1024² 分辨率下 Token 数平方增长对比图
    |-- p4_pipeline.png                             # 预览稿 P4: VLM 三框流水线示意图（高亮视觉编码器并嵌入 FastViT 结构）
    |-- p11_token_grid_coarse.png                   # 预览稿 P10: 下采样网格变粗导致细小文字丢失的直观示意图
    |-- p13_channel_stack.png                       # 预览稿 P12: “空间换通道”特征图尺寸与通道维度的直观解释图
    |-- p14_multiscale_concat.png                   # 预览稿 P13: 多尺度特征经 Learned Pool 对齐并在通道维拼接（C）示意图
    |-- p15_same_tokens_resolution.png              # 预览稿 P14: 相同 256 个 Token 下 ViT (224²) vs. FastViTHD (1024²) 清晰度对比
    |-- zoomA_fastvit_stage_column.png              # 预览稿 P6 动画底图: FastViT 左侧 4 阶段金字塔主干放大图（无标注）
    |-- zoomA_fastvit_stage_column_annotated.png    # 预览稿 P6: FastViT 左侧 4 阶段主干放大标注图（4x~32x 与 1024 Token）
    |-- zoomB_fastvit_stage4_attention.png          # 预览稿 P7: FastViT Stage 4 自注意力模块局部放大图
    |-- zoomC_fastvit_repmixer.png                  # 预览稿 P5: FastViT RepMixer 训练态多分支与推理态单卷积合成放大图
    `-- src/                                        # 自绘示意图的 HTML 源码（使用 Chrome 无头模式渲染截图）
        |-- channel_stack.html                      # 生成 p13_channel_stack.png 的 HTML 源码
        |-- fastvit_column_noimg.png                # FastViT 阶段列去插图底图素材
        |-- input_photo.png                         # 流水线与网格示意图使用的示例输入图像素材
        |-- multiscale_concat.html                  # 生成 p14_multiscale_concat.png 的 HTML 源码
        |-- pipeline.html                           # 生成 p4_pipeline.png 的 HTML 源码
        |-- sample_doc.html                         # 生成文档清晰度对比底图的 HTML 源码
        |-- sample_doc_1024.png                     # 渲染出的 1024x1024 示例文档底图
        `-- token_count_chart.html                  # 生成 p2_token_count_chart.png 的 HTML 源码
```

## 文档导航

| 文档 | 说明 |
|---|---|
| [01-pre-outline.md](01-pre-outline.md) | 7 分钟展示定稿大纲：围绕「高分辨率慢在哪 $\rightarrow$ 前作 FastViT 结构 $\rightarrow$ 哪里压得不够 $\rightarrow$ FastViTHD 怎么再压一层且不丢细节 $\rightarrow$ 训练与系统代价」展开。 |
| [02-slides.md](02-slides.md) | 逐页幻灯片设计规范：定义「一页一问题、一页一主图、正文不超三行、术语与配色统一」等全局约束及逐页配图方案。 |
| [03-slides-preview.md](03-slides-preview.md) | 17 页 Marp 幻灯片完整源码，包含每页版式与 HTML 注释中的逐页口播讲稿。 |
| [03-slides-preview.pdf](03-slides-preview.pdf) | 由 [03-slides-preview.md](03-slides-preview.md) 导出的 16:9 幻灯片 PDF 预览文件。 |

## 幻灯片配图速览

| 预览稿页码 | 图文件 | 画面内容与用途 |
|---|---|---|
| **P2** | [p2_token_count_chart.png](figures/p2_token_count_chart.png) | 对比 ViT 在 $336^2$（576 tokens）与 $1024^2$（5350 tokens）下的 Token 暴增。 |
| **P4** | [p4_pipeline.png](figures/p4_pipeline.png) | VLM 三框流水线（视觉编码器 $\rightarrow$ 投影层 $\rightarrow$ LLM），标明 FastVLM 仅替换视觉编码器。 |
| **P5** | [zoomC_fastvit_repmixer.png](figures/zoomC_fastvit_repmixer.png) | 放大展示 `RepMixer` 训练态多分支到推理态单卷积的结构重参数化。 |
| **P6** | [zoomA_fastvit_stage_column.png](figures/zoomA_fastvit_stage_column.png) / [标注版](figures/zoomA_fastvit_stage_column_annotated.png) | 放大展示 FastViT 4 阶段金字塔下采样（$4\times \rightarrow 32\times$），标出 $1024^2$ 下仍剩 1024 个 Token。 |
| **P7** | [zoomB_fastvit_stage4_attention.png](figures/zoomB_fastvit_stage4_attention.png) | 放大展示自注意力模块，配合表格说明在 $16\times$ 下采样处放大模型会导致大图自注意力开销激增 400 倍。 |
| **P9** | [fastvit_stage_column_annotated_p7.png](figures/fastvit_stage_column_annotated_p7.png) / [fastvithd_stage_column_annotated.png](figures/fastvithd_stage_column_annotated.png) | 将 4 阶段 FastViT（压 $32\times$）与 5 阶段 FastViTHD（压 $64\times$，自注意力后移至 $32\times/64\times$）等高并排对比。 |
| **P10** | [p11_token_grid_coarse.png](figures/p11_token_grid_coarse.png) | 直观展示下采样越狠、单个 Token 覆盖网格越粗，容易丢失文档小字细节。 |
| **P11** | [fastvithd_stages_p9_annotated.png](figures/fastvithd_stages_p9_annotated.png) | 在 FastViTHD 横版架构图上标出保证不丢细节的三项设计：①先看细节再压缩、②空间换通道（96$\rightarrow$1536）、③多尺度特征。 |
| **P12** | [p13_channel_stack.png](figures/p13_channel_stack.png) | 解释「什么是通道」以及空间长宽减半、通道数翻倍至 1536 维的信息保留机制。 |
| **P13** | [p14_multiscale_concat.png](figures/p14_multiscale_concat.png) / [原图裁剪](figures/fastvithd_multiscale_branches.png) | 解释多尺度支路如何通过 `Learned Pool` 对齐到 $16\times 16$ 并沿通道方向拼接（`C`）。 |
| **P14** | [p15_same_tokens_resolution.png](figures/p15_same_tokens_resolution.png) | 对比同样 256 个 Token 预算下，ViT 只能看模糊的 $224^2$ 小图，而 FastViTHD 能看清晰的 $1024^2$ 大图。 |
