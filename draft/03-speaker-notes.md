# FastVLM speaker notes

Extracted from the `<!-- -->` notes in [03-slides-preview.md](03-slides-preview.md), one section per slide.

## 1. FastVLM

*(no notes)*

## 2. Higher resolution, slower VLM

Say: Pick up from the previous speaker. A ViT never downsamples: the bigger the image, the more tokens, and every layer runs self-attention over all of them.

## 3. Cutting tokens is not enough

Say: First make clear this chart shows the paper's own model: its token count is already small, yet encoding still takes most of the time. So cutting tokens alone is not enough; tiling is even worse, because it runs the encoder more times.

## 4. FastVLM swaps one box in the pipeline

*(no notes)*

## 5. Predecessor FastViT: the big picture

Say: Walk through the overall layout on the figure.
- The left column is the backbone: the stem downsamples 4×, then come 4 stages, with one more downsampling between each pair of stages.
- The first 3 stages use convolution (cheap, local view); the last one uses self-attention (expensive, global view).
- On the right, (a)–(d) show the inside of each block. Next we zoom in piece by piece: first the conv part, then where the design breaks down at high resolution.

## 6. The conv part is fine: one conv at inference

Say: This is RepMixer's structural reparameterization. During training, y = conv(x) + x, with a skip branch; after training, since every operation is linear, it can be merged into one equivalent conv. The output is the same, but inference saves one round of memory reads and writes, which pays off most on large images.
Transition: The conv part is fine, so where is the problem? Look at the other two places.

## 7. Downsampled 32×: why is it still not enough?

Animation: First show the stage column without labels (figures/zoomA_fastvit_stage_column.png) and explain the pyramid; then bring in the downsampling labels and the red box and explain why it is still not enough.
Say: Contrast with patchify: a ViT cuts the image into patches once and the token count never changes; a pyramid keeps shrinking the map as it computes.

## 8. Self-attention comes too early

Say: A four-step chain: we want a bigger model → the habit is to stack self-attention in stages 3 and 4 → those designs were made for small images → switch to large images and the cost explodes.
Transition: That is just on paper. How slow is it when it actually runs?

## 9. In practice: scaled the usual way, slower than pure conv

Say:
- Note that the y-axis is on a log scale.
- The orange line is the approach from the last slide: stacking self-attention in stages 3 and 4. At high resolution it is slower than even the pure-conv green line.
- So the root cause is confirmed: not enough compression. Too little before self-attention, and compute explodes; too little before the LLM, and there are too many tokens.
- The blue line has the same parameter count as the orange one, yet it is much faster. How? Next slide.

## 10. Compress one more level

Say: One change fixes both problems at once. This is the blue line from the last slide.

## 11. Why didn't anyone do this earlier?

*(no notes)*

## 12. Compressing the right way

*(no notes)*

## 13. ② Trading space for channels: what is a "channel"?

Say: Channels are not layers. Layers count how many operations are stacked (FastViTHD's five stages have 2, 12, 24, 4 and 2 layers); channels count how many numbers describe each position. An RGB image already has 3 channels. Each position in the last stage is one token, and it carries 1536 numbers.

## 14. ③ Multi-scale: what is the C?

Alternative figure: a crop of the paper's original figure, figures/fastvithd_multiscale_branches.png.
Say:
- Earlier stages have larger maps and see finer detail; the last stage is the most condensed and the most "semantic". Pulling the earlier stages in brings back detail lost during compression.
- Learned Pool vs. ordinary pooling: ordinary pooling just averages each small block; Learned Pool shrinks the map with a trainable conv. The paper tried both, and the learned one is slightly better.
- C (concatenation): at each position, the numbers from the branches are joined end to end, so the channel vector gets longer; the number of positions does not change, so neither does the token count.
- The projector then maps this long vector to the length the LLM can read.

## 15. Compression buys resolution

Say: This is the same crop of the same image. Each red cell is one token, and on both sides the whole image is 16×16 = 256 tokens. On the left, 256 tokens only let a ViT see a 224² image, and the small text is already a blur; on the right, FastViTHD sees 1024², so both the table and the note are readable. The sample is fictional data we made ourselves.

## 16. Cost ①: harder compression needs stronger training

Say:
- This figure is the training-data pipeline from their earlier MobileCLIP, which FastViTHD reuses.
- (a) Augment each image, then have a captioning model write better captions;
- (b) have several very large CLIP models (the teachers) encode both the image and the text;
- (c) store all of this in the dataset, so the small model learns from the teachers' outputs during training, which is distillation.
- This is our reading: keeping its visual ability under such heavy compression very likely depends on this training.

## 17. Cost ②: resolution must match the LLM size

*(no notes)*
