# FastVLM speaker notes

Extracted from the `<!-- -->` notes in [03-slides-preview.md](03-slides-preview.md), one section per slide.

## 1. FastVLM

*(no notes)*

## 2. Higher resolution, slower VLM

Say:
- TTFT, time to first token, is how long we wait before the model gives its first word. It has two parts: the vision encoder turns the image into tokens, then the LLM reads all of them (prefill).
- Higher resolution means more tokens: triple the side, nine times the tokens.
- More tokens slow down every module that processes them. Self-attention is the worst: every token looks at every other token, so its cost grows with the square of the token count. This hits both the encoder and the LLM.

## 3. Trimming tokens is not enough

Say:
- This chart splits TTFT into the two parts from the last slide. Note the y-axis is on a log scale.
- As resolution grows, the orange bar (encoding) catches up with the blue bar (prefill) and then overtakes it.
- Token-pruning methods leave the encoder as it is and trim tokens after it. At best that removes the blue bar; the orange bar stays, because the encoder has already processed the whole image.
- So the bottleneck is the encoder itself.
If asked: The chart is measured on the paper's own, already efficient encoder; a standard encoder would be even slower.

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

## 13. Remember this from class?

Say:
- Remember this CNN from class. The red box is the feature-learning part: conv and pool layers shrink the map step by step, while the number of channels grows.
- FastViTHD's stages are exactly this part. The FC layers for classification are not used: in a VLM, the final feature map goes to the projector and the LLM instead.
- Next: FastViTHD uses the same idea.

## 14. ② Trading space for channels

Say: Channels are not layers. Layers count how many operations are stacked (FastViTHD's five stages have 2, 12, 24, 4 and 2 layers); channels count how many numbers describe each position. An RGB image already has 3 channels. Each position in the last stage is one token, and it carries 1536 numbers.

## 15. ③ Multi-scale: what is the C?

Alternative figure: a crop of the paper's original figure, figures/fastvithd_multiscale_branches.png.
Say:
- Earlier stages have larger maps and see finer detail; the last stage is the most condensed and the most "semantic". Pulling the earlier stages in brings back detail lost during compression.
- Learned Pool vs. ordinary pooling: ordinary pooling just averages each small block; Learned Pool shrinks the map with a trainable conv. The paper tried both, and the learned one is slightly better.
- C (concatenation): at each position, the numbers from the branches are joined end to end, so the channel vector gets longer; the number of positions does not change, so neither does the token count.
- The projector then maps this long vector to the length the LLM can read.

## 16. Compression buys resolution

Say: This is the same crop of the same image. Each red cell is one token, and on both sides the whole image is 16×16 = 256 tokens. On the left, 256 tokens only let a ViT see a 224² image, and the small text is already a blur; on the right, FastViTHD sees 1024², so both the table and the note are readable. The sample is fictional data we made ourselves.

## 17. Cost ①: harder compression needs stronger training

Say:
- This figure is the training-data pipeline from their earlier MobileCLIP, which FastViTHD reuses.
- (a) Augment each image, then have a captioning model write better captions;
- (b) have several very large CLIP models (the teachers) encode both the image and the text;
- (c) store all of this in the dataset, so the small model learns from the teachers' outputs during training, which is distillation.
- This is our reading: keeping its visual ability under such heavy compression very likely depends on this training.

## 18. Cost ②: resolution must match the LLM size

*(no notes)*

## 19. Result ①: same accuracy, 3.2× faster

Say:
- This is the paper's Fig. 1. Every encoder is paired with the same 0.5B LLM and trained the same way, so only the encoder differs.
- The small labels are input resolutions. Follow the blue line: from 256² to 1024², accuracy keeps rising while the time stays under 0.2 s.
- SigLIP, a ViT-style encoder, needs about half a second just for 384². The 3.2× arrow is the paper's own.

## 20. Result ②: one whole image beats tiles

Say:
- This answers the "image tiling" patch from the start of the talk. Note the x-axis is on a log scale.
- Compare points at the same resolution, for example 1024²: the blue square is higher and further left than every pink 1024² point.
- Tiling was a workaround for encoders that cannot take large images. FastViTHD can, so it does not need it; only at 1536² do 2×2 tiles draw level.

## 21. Result ③: vs. a tiling model with the same LLM

Say:
- This is the paper's headline comparison. Both models use the same 0.5B LLM.
- LLaVA-OneVision cuts a 1152² image into tiles and ends up with 7,290 tokens, so most of its 14 seconds is the LLM reading tokens.
- FastVLM sends 256 tokens from one 1024² image: 0.17 s in total, and it still scores as well or better on SeedBench, MMMU and DocVQA.

## 22. Discussion

Say:
- Two things the results support, and two to keep in mind.
- The cleanest evidence is Result ①, where the training is identical. Result ③ is the most striking number, but there the training data also differs (paper Table 6, rows R2 and R4).
- On the device: the paper runs the encoder on the Mac's neural engine and the LLM on its GPU, so the speed-up on a server GPU could be different.
