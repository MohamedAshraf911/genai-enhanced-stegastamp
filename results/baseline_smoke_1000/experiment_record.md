# StegaStamp baseline smoke test

## Setup

- Environment: Python 3.11, TensorFlow 2.15.1
- Training subset: 500 images selected with seed 42, as recorded in
  `data/coco/subset_500/manifest.json`
- Training run: 1,000 steps, secret size 100, standard training options,
  completed in about 19 minutes
- Model: newly trained exploratory model at
  `saved_models/baseline_smoke_1000`, not the official pretrained model
- Cover selection: first three filenames in the seed-42 manifest
- Secret: `Hello`

The training run produced checkpoint
`checkpoints/baseline_smoke_1000/baseline_smoke_1000.chkp-1000` and an
exported SavedModel. The original `saved_models/stegastamp_pretrained` was
not available.

## Results

The decoder returns the message padded to seven bytes. Padding spaces are
shown in parentheses below; decoded messages match after removing that
right-padding.

| # | Cover image | Secret | Decoder output | MAE | PSNR | Result |
|---|-------------|--------|----------------|-----|------|--------|
| 1 | `000000107554.jpg` | `Hello` | `Hello` (2 padding spaces) | 0.2017 | 11.63 dB | Decode pass; visual similarity fail |
| 2 | `000000021167.jpg` | `Hello` | `Hello` (2 padding spaces) | 0.1926 | 11.59 dB | Decode pass; visual similarity fail |
| 3 | `000000260657.jpg` | `Hello` | `Hello` | 0.2302 | 10.50 dB | Decode pass; visual similarity fail |

MAE and PSNR compare each encoder-fitted 400x400 cover against its
watermarked output. They are simple pixel-difference checks, not perceptual
metrics.

## Artifacts

- [Comparison figure](./comparison.png)
- [First cover](./original.png)
- [First watermarked image](./watermarked.png)
- [First residual](./residual.png)
- All three covers and outputs are also preserved as
  `<image-id>_original.png`, `<image-id>_watermarked.png`, and
  `<image-id>_residual.png` in this directory.
- Raw encoder outputs are in `../../out/baseline_smoke_1000/`.

## Conclusion

The 1,000-step exploratory model completes training, exports a SavedModel,
and decodes `Hello` on these three images. However, the comparison figure
shows severe visible block artifacts and the low PSNR values confirm large
cover changes. This is a pipeline smoke test, **not a successful visual
reproduction of the pretrained StegaStamp baseline**. Obtain the official
pretrained model or run the approved full training budget before claiming an
M1 reproduction.
