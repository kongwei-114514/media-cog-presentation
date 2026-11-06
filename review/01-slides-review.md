# 幻灯片草稿审阅记录（22 页英文版 `draft/03-slides-preview`）

对应审阅版本：[`draft/03-slides-preview.md`](../draft/03-slides-preview.md) / [`draft/03-slides-preview.pdf`](../draft/03-slides-preview.pdf) / [`draft/03-speaker-notes.md`](../draft/03-speaker-notes.md)（共 22 页）

---

## 1. 核心图文不匹配问题：P8「Self-attention comes too early」

### 现象描述
第 8 页（`# Self-attention comes too early`）左侧配图 [`zoomB_fastvit_stage4_attention.png`](../draft/figures/zoomB_fastvit_stage4_attention.png) 与右侧文字及表格所讲内容不在同一维度，图文之间缺乏对应关系。

### 原因拆解
1. **右侧文字讲的是「宏观阶段位置与特征图尺寸（Stage-level）」**：
   - 文字指出：将 4 阶段混合模型放大时（对应 `FastViT-Naive Scaling`），习惯在 **Stage 3（16× down）** 和 Stage 4 多堆自注意力层；
   - 表格对比了 **Stage 3（16× down）** 下 $224^2$ 小图（$14\times 14 = 196$ 个位置）与 $1024^2$ 大图（$64\times 64 = 4096$ 个位置）的位置数差异，说明自注意力开销随位置数平方增长（贵约 $400\times$）。
2. **左侧配图画的却是「单个自注意力块内部的算子重参数化（Operator-level）」**：
   - [`zoomB_fastvit_stage4_attention.png`](../draft/figures/zoomB_fastvit_stage4_attention.png) 裁剪自第 5 页 `FastViT` 原图（[`fastvit_arch.png`](../paper/figures/fastvit_arch.png)）中间 `(a)` 的底部米色框；
   - 原图中该框左右两列分别对应 **训练态（Training）** 与 **推理态（Inference）**，中间箭头表示条件位置编码（`CPE`）将旁路 `7x7 DW Conv` 合并进单层的结构重参数化过程；
   - 裁剪后不仅丢失了顶部的 `Training / Inference` 表头（导致左右两列与中间箭头含义不明），而且图中的 `CPE`、`BatchNorm`、`ConvFFN` 等微观结构与右侧文字讨论的「Stage 3 下采样 16 倍、位置数从 196 暴增至 4096」毫无关联。
3. **与前一页（P7）存在视觉断层**：
   - 在 P7（[`zoomA_fastvit_stage_column_annotated.png`](../draft/figures/zoomA_fastvit_stage_column_annotated.png)）中，展示的原版 `FastViT` 的 **Stage 3 是灰色框（Conv）**，只有 Stage 4 是米色框（Attention）；
   - 到了 P8，文字直接开始计算「Stage 3（16× down）做自注意力」的开销，但画面上并未展示「模型放大时把 Stage 3 也改成了自注意力」。

---

## 2. 跨页脉络割裂问题：P12 $\rightarrow$ P14、P15（及 P16，「Compressing the right way」主线连续性）

### 现象描述
第 12 页（`# Compressing the right way`）放出了标注有 `① Details first, then compress`、`② Channels`、`③ Multi-scale features` 的 `FastViTHD` 横版核心架构图（[`fastvithd_stages_p9_annotated.png`](../draft/figures/fastvithd_stages_p9_annotated.png)），并将 ② 和 ③ 拆到了后续页面展开。但在经过第 13 页（`# Remember this from class?`，课程 CNN 回顾 [`course_cnn_feature_learning.png`](../draft/figures/course_cnn_feature_learning.png)）翻到 **P14（`# ② Trading space for channels`）** 和 **P15（`# ③ Multi-scale: what is the C?`）** 时，P12 这张总览架构图完全消失，导致听众容易忘记当前正在回答什么总问题、也看不出当前讲的局部机制对应架构图里的哪个位置，整条思路被拆散割裂。

### 具体断层表现
1. **局部解释脱离了 P12 的架构总图锚点（插入 P13 后跨度更远）**：
   - P12 在架构图上标出了 ①、②、③ 三个位置，并预告后续页面展开 ② 和 ③（注：因中间插入了 P13 课程回顾页，P12 左栏写的 `② ... (next slide)` 与 `③ ... (the slide after)` 实际已顺延至 P14 与 P15）；
   - 当翻过 P13 到达 P14 和 P15 时，画面全部换成了抽象的三维长方体示意图（[`p13_channel_stack.png`](../draft/figures/p13_channel_stack.png)、[`p14_multiscale_concat.png`](../draft/figures/p14_multiscale_concat.png)），P12 的横版架构图不再出现；
   - 尤其是 **P15 标题直接问 `③ Multi-scale: what is the C?`**，但带有 `C` 节点的原架构图远在三页前的 P12，听众在 P15 画面上根本看不到原架构图里的分支和 `C` 在哪。
2. **连续展开概念后容易忘掉主线问题（P11/P12 $\rightarrow$ P14/P15/P16）**：
   - P11～P16 本质上是一条完整的逻辑线：回答 **「压 64 倍为什么不会把细节丢光，反而看得更清？」**；
   - 因为中间连续插入了课程 CNN 回顾（P13）、通道尺寸变化（P14）、Learned Pool 与拼接 C（P15），如果 P14、P15 页面上没有 P12 的架构图做串联，听众很容易陷入局部概念解释，忘了这几页合在一起是在回答「`FastViTHD` 怎么压 64 倍还不丢细节」这个核心问题。
3. **后续改进方向记录**：
   - 在拆解页（P14、P15，以及收束页 P16）保留或同步展示 P12 的 `FastViTHD` 横版架构图（将其作为同页视觉锚点/局部高亮对照），让「总架构位置 $\leftrightarrow$ 局部机制原理 $\leftrightarrow$ 最终分辨率收益」始终处于同一条视觉主线上。

---

## 3. 逻辑未讲透、归因错位与前后衔接：P18「Cost ②: resolution must match the LLM size」

### 现象描述
第 18 页（`# Cost ②: resolution must match the LLM size`）目前写了两条要点（`A very small LLM can't make use of ultra-high-resolution images`、`However well the vision side compresses, both ends still have to be tuned together`），并在页面底部放了一句红字：`Redesign the encoder from the inside, instead of patching it from the outside`。

### 问题分析
1. **核心意思只说了一半，缺少关键转折**：
   - 提升视觉编码器（高分辨率输入 + 高密度压缩 Token）确实是以**对后端 LLM 提出更高能力要求**为代价的（小尺寸 LLM 无法充分消化超高分辨率带来的高密度视觉信息，必须搭配更大规模的 LLM 才能完全发挥潜力）；
   - 但这一页真正要传达的核心重点（对应论文 Sec. 3.2.1 与 Fig. 4 [`decoder_encoder_ablation.png`](../paper/figures/decoder_encoder_ablation.png)）是：**即便需要 LLM 规模配合，升级视觉编码器相比于「只提升 LLM 而不改编码器」依然具有压倒性的系统优势（整条帕累托前沿显著上移、同精度下快 3 倍）**。现有文案只停留在「小 LLM 用不上、两头得一起调」，没有把「改编码器依然比单纯堆大 LLM 划算得多」这一核心优势落下来。
2. **底部红字归因错位，且与新增的 P22 结语重复**：
   - 页面最下方的红字 `Redesign the encoder from the inside, instead of patching it from the outside` 原本是 17 页旧版作为全篇收尾的口号，针对的是外部切图/剪枝方案，与本页上方讨论的「Encoder 与 LLM 的协同关系」并非因果关系；
   - 在 22 页新版中，后面新增了 P19～P21（`Result ①～③`）和 P22（`# Discussion`），且 P22 底部已经有了真正的全篇收尾结论（`→ Fixing the encoder itself is what makes high resolution affordable`）。这句旧结语留在 P18 底部既导致归因错位，又在半途提前喊了一次口号。
3. **与下一页 P19（`Result ①`）存在表面表述冲突**：
   - P18 刚断言「很小的 LLM 用不上高分辨率图像」，紧接着翻到 **P19（`# Result ①: same accuracy, 3.2× faster`）**，展示的论文 Fig. 1(a)（[`acc_vs_latency_qwen.png`](../paper/figures/acc_vs_latency_qwen.png)）却恰恰固定使用 **0.5B 的小 LLM**，且讲稿强调从 $256^2$ 到 $1024^2$ 精度持续上升；
   - 实际上 0.5B LLM 是在超过 $1024^2$ 的极端高分辨率（$1536^2 / 2048^2$，见 Fig. 4）下才饱和躺平，且换成 7B 模型能把精度上限从 ~55% 进一步推高到 ~65%。若 P18 不加区分地写“小 LLM 用不上高分辨率”，听众翻到 P19 看到 0.5B 模型在 $1024^2$ 下的亮眼表现会感到困惑。

### 更改思路（按论文 Sec. 3.2.1 主线理顺）
- 先讲清**代价**：编码器推向超高分辨率与高压缩后，信息密度更高，小 LLM 在更高分辨率（$>1024^2$）下会提前饱和，需要更大容量的 LLM 配合才能吃满全部增益；
- 再落脚到**核心优势**：对比「升级编码器 + 匹配 LLM」与「不改编码器、只靠放大 LLM」，明确指出改编码器能够把整个 VLM 系统的精度-延迟帕累托上限全面抬高（自然引出后面的 Result ①～③）；
- 移除 P18 底部与本页内容无关的旧结语红字，将该页结论收束在「视觉编码器对整个 VLM 效率上限的决定作用」上。

---

## 4. 其他版式与细节观察（备查）

### 4.1 渲染版式细节
- **P10（「Compress one more level」）左侧双图并排问题**：
  - 使用 Marp 双背景图并排放置 [`fastvit_stage_column_annotated_p7.png`](../draft/figures/fastvit_stage_column_annotated_p7.png) 与 [`fastvithd_stage_column_annotated.png`](../draft/figures/fastvithd_stage_column_annotated.png) 时，左列紧贴画面最左边缘（无留白）；
  - 两列顶部仅写了 `Stem`，缺少标明「左：`FastViT`（4 stages） vs. 右：`FastViTHD`（5 stages）」的列标题。
- **分栏页面的大标题折行**：
  - **P15**（`# ③ Multi-scale: what is the C?`）、**P19**（`# Result ①: same accuracy, 3.2× faster`）、**P20**（`# Result ②: one whole image beats tiles`）由于配图占用了 `52%～58%` 的宽度，文字侧的标题被挤成了两行。
- **画面留白平衡**：
  - **P2**：柱状图 [`p2_token_count_chart.png`](../draft/figures/p2_token_count_chart.png) 默认靠左，页面右侧留白偏多。
  - **P8**：左图黑框紧贴幻灯片左边缘，缺少左边距。

### 4.2 叙事呼应与文档索引备忘
- **P7 与 P8 的呼应**：P7（交给 LLM 的 Token 太多，对外瓶颈）与 P8（自注意力放太早，对内编码瓶颈）正好对应 P2～P3 拆解的两部分延迟，可在标题或批注中强化「问题① / 问题②」的对应感，以便在 P10（「One change fixes both problems at once」）形成更强收束。
- **仓库 README 索引同步**：当前 `draft/` 已更新为 22 页英文版，并新增了 [`03-speaker-notes.md`](../draft/03-speaker-notes.md)、[`03-slides-preview.html`](../draft/03-slides-preview.html)、[`03-slides-preview.pptx`](../draft/03-slides-preview.pptx) 及两张新配图（[`course_cnn_feature_learning.png`](../draft/figures/course_cnn_feature_learning.png)、[`p20_ttft_vs_onevision.png`](../draft/figures/p20_ttft_vs_onevision.png)），后续定稿时可同步更新根目录 [`README.md`](../README.md) 与 [`draft/README.md`](../draft/README.md) 中的页数与文件清单。
