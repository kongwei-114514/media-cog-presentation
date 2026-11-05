---
marp: true
lang: en
theme: default
paginate: true
size: 16:9
---

<style>
section { font-size: 28px; padding: 48px 60px; }
h1 { font-size: 42px; color: #1f2937; margin-bottom: 0.4em; }
strong { color: #D62728; }
table { font-size: 24px; }
blockquote { border: 2px dashed #999; border-radius: 8px; padding: 0.4em 0.8em; color: #666; font-size: 22px; }
section.lead { justify-content: center; text-align: center; }
section.lead h1 { font-size: 64px; }
section.dense { font-size: 24px; }
section.center p { text-align: center; }
</style>

<!-- _class: lead -->
<!-- _paginate: false -->

# FastVLM

Redesigning the vision encoder for high resolution

CVPR 2025 · Apple

---

# Higher resolution, slower VLM

**TTFT = vision encoding time + LLM prefill time**

![h:390](figures/p2_token_count_chart.png)

Token count grows with the **square** of resolution

<!--
Say: Pick up from the previous speaker. A ViT never downsamples: the bigger the image, the more tokens, and every layer runs self-attention over all of them.
-->

---

# Cutting tokens is not enough

![bg right:48% contain](../paper/figures/ttft_breakdown.png)

- At high resolution, **vision encoding itself** takes most of the time
- Token pruning, image tiling: patches applied outside the ViT

→ Can we use an encoder **built for high resolution**?

<!--
Say: First make clear this chart shows the paper's own model: its token count is already small, yet encoding still takes most of the time. So cutting tokens alone is not enough; tiling is even worse, because it runs the encoder more times.
-->

---

# FastVLM swaps one box in the pipeline

![h:390](figures/p4_pipeline.png)

- Only the vision encoder changes; projector and LLM follow LLaVA
- Every stage from here on lives **inside** this box

---

<!-- _class: center -->

# Predecessor FastViT: the big picture

![h:470](../paper/figures/fastvit_arch.png)

Left: the skeleton, a 4-stage pyramid | Right: the parts inside each block

<!--
Say: Walk through the overall layout on the figure.
- The left column is the backbone: the stem downsamples 4×, then come 4 stages, with one more downsampling between each pair of stages.
- The first 3 stages use convolution (cheap, local view); the last one uses self-attention (expensive, global view).
- On the right, (a)–(d) show the inside of each block. Next we zoom in piece by piece: first the conv part, then where the design breaks down at high resolution.
-->

---

# The conv part is fine: one conv at inference

![h:380](figures/zoomC_fastvit_repmixer.png)

Training: several branches (easy to train) → Inference: merged into one conv (fast)
The bigger the image, the more it saves → **suits high resolution, keep as is**

<!--
Say: This is RepMixer's structural reparameterization. During training, y = conv(x) + x, with a skip branch; after training, since every operation is linear, it can be merged into one equivalent conv. The output is the same, but inference saves one round of memory reads and writes, which pays off most on large images.
Transition: The conv part is fine, so where is the problem? Look at the other two places.
-->

---

# Downsampled 32×: why is it still not enough?

![bg right:35% contain](figures/zoomA_fastvit_stage_column_annotated.png)

- Each stage downsamples once; **every position left at the end is one token**
- 1024² image → 32×32 = **1024** tokens, still too many for the LLM

→ Can we **compress further**?

<!--
Animation: First show the stage column without labels (figures/zoomA_fastvit_stage_column.png) and explain the pyramid; then bring in the downsampling labels and the red box and explain why it is still not enough.
Say: Contrast with patchify: a ViT cuts the image into patches once and the token count never changes; a pyramid keeps shrinking the map as it computes.
-->

---

# Self-attention comes too early

![bg left:40% contain](figures/zoomB_fastvit_stage4_attention.png)

To scale a model up, the usual move is stacking self-attention in stages 3 and 4

| Stage 3 (16× down) | Positions | Self-attention cost |
|---|---|---|
| 224² image | 196 | cheap |
| 1024² image | 4096 | about **400×** more |

<!--
Say: A four-step chain: we want a bigger model → the habit is to stack self-attention in stages 3 and 4 → those designs were made for small images → switch to large images and the cost explodes.
Transition: That is just on paper. How slow is it when it actually runs?
-->

---

<!-- _class: center -->

# In practice: scaled the usual way, slower than pure conv

![h:430](../paper/figures/resolution_scaling_trend.png)

Orange: usual scaling | Green: pure-conv ConvNeXt-L | Blue: FastViTHD (next slide)

<!--
Say:
- Note that the y-axis is on a log scale.
- The orange line is the approach from the last slide: stacking self-attention in stages 3 and 4. At high resolution it is slower than even the pure-conv green line.
- So the root cause is confirmed: not enough compression. Too little before self-attention, and compute explodes; too little before the LLM, and there are too many tokens.
- The blue line has the same parameter count as the orange one, yet it is much faster. How? Next slide.
-->

---

# Compress one more level

![bg left:55% contain](figures/fastvit_stage_column_annotated_p7.png)
![bg contain](figures/fastvithd_stage_column_annotated.png)

- Add a 5th stage: **64×** downsampling in total
- Move self-attention to the 32× and 64× stages

| ViT | FastViT | FastViTHD |
|---|---|---|
| ≈ 5300 | 1024 | **256** |

(tokens for a 1024² input)

<!--
Say: One change fixes both problems at once. This is the blue line from the last slide.
-->

---

# Why didn't anyone do this earlier?

- The harder you compress, the more area one token covers → **small text gets lost**
- The view at the time: ViTamin recommends only 16× downsampling in total

→ The real question: **how do we compress without losing information?**

![w:1050](figures/p11_token_grid_coarse.png)

---

# Compressing the right way

![bg right:58% contain](figures/fastvithd_stages_p9_annotated.png)

① Details first, then compress
Cheap convs (RepMixer) while the map is large; self-attention only once it is small

② Trade space for channels (next slide)

③ Multi-scale features (the slide after)

---

<!-- _class: dense -->

# ② Trading space for channels: what is a "channel"?

- Feature map = height × width × **channels**
- Each conv kernel outputs one channel; more kernels, more channels → **how many features** each position has

![w:1000](figures/p13_channel_stack.png)

<!--
Say: Channels are not layers. Layers count how many operations are stacked (FastViTHD's five stages have 2, 12, 24, 4 and 2 layers); channels count how many numbers describe each position. An RGB image already has 3 channels. Each position in the last stage is one token, and it carries 1536 numbers.
-->

---

<!-- _class: dense -->

# ③ Multi-scale: what is the C?

![bg right:58% contain](figures/p14_multiscale_concat.png)

1. Take one branch each from stages 2, 3 and 4
2. **Learned Pool**: a strided depthwise conv shrinks each to stage 5's size (16×16)
3. **C**: concatenate the branches along the **channel** dimension

→ Still **256** tokens, but each token now carries both coarse and fine information

<!--
Alternative figure: a crop of the paper's original figure, figures/fastvithd_multiscale_branches.png.
Say:
- Earlier stages have larger maps and see finer detail; the last stage is the most condensed and the most "semantic". Pulling the earlier stages in brings back detail lost during compression.
- Learned Pool vs. ordinary pooling: ordinary pooling just averages each small block; Learned Pool shrinks the map with a trainable conv. The paper tried both, and the learned one is slightly better.
- C (concatenation): at each position, the numbers from the branches are joined end to end, so the channel vector gets longer; the number of positions does not change, so neither does the token count.
- The projector then maps this long vector to the length the LLM can read.
-->

---

# Compression buys resolution

![w:1100](figures/p15_same_tokens_resolution.png)

The same 256 tokens: for a VLM, **resolution is what matters**

<!--
Say: This is the same crop of the same image. Each red cell is one token, and on both sides the whole image is 16×16 = 256 tokens. On the left, 256 tokens only let a ViT see a 224² image, and the small text is already a blur; on the right, FastViTHD sees 1024², so both the table and the note are readable. The sample is fictional data we made ourselves.
-->



---

<!-- _class: center -->

# Cost ①: harder compression needs stronger training

![w:1100](../paper/figures/mobileclip_dr_schematic.png)

Each token must hold more information → pretraining has to be stronger
Taken from their MobileCLIP: prepare **better captions** and **large teacher models' outputs** for every image-text pair

<!--
Say:
- This figure is the training-data pipeline from their earlier MobileCLIP, which FastViTHD reuses.
- (a) Augment each image, then have a captioning model write better captions;
- (b) have several very large CLIP models (the teachers) encode both the image and the text;
- (c) store all of this in the dataset, so the small model learns from the teachers' outputs during training, which is distillation.
- This is our reading: keeping its visual ability under such heavy compression very likely depends on this training.
-->

---

# Cost ②: resolution must match the LLM size

- A very small LLM **can't make use of** ultra-high-resolution images
- However well the vision side compresses, both ends still have to be tuned together

<br>

**Redesign the encoder from the inside, instead of patching it from the outside**
