# NewStegaStamp

This project carries the original StegaStamp encoder/decoder and training
workflow into a modern TensorFlow 2 compatibility environment. The original
`StegaStamp` directory is left unchanged.

## Requirements

- Windows 10/11, 64-bit
- Python 3.11
- TensorFlow 2.15.1 (CPU on native Windows)
- A training image directory if training from scratch

From PowerShell in this directory, install `uv` using your existing Python,
then create the project environment:

```powershell
python -m pip install uv
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

TensorFlow 2.15.1 retains the graph/session APIs used by the original code.
`tf_compat.py` enables TensorFlow 1.x graph behavior, and the removed
`tensorflow.contrib` image transforms and third-party spatial transformer are
implemented in `utils.py`.

## Train

Place training images in a directory, then run:

```powershell
.\.venv\Scripts\python.exe train.py my_experiment --data_dir .\data\images --secret_size 100
```

Training downloads the LPIPS TensorFlow graph on first use. It requires an
internet connection then. Training checkpoints are saved under `checkpoints`,
TensorBoard logs under `logs`, and the exported SavedModel under
`saved_models\my_experiment`.

Use `--secret_size 100` for compatibility with the included message
encoder/decoder (96 BCH-protected bits plus 4 padding bits). The training
script's default secret size is retained for reproducing the original
training setup, but it does not match the message encoding scripts.

## Prepare a reproducible COCO subset

Extract COCO `val2017` into `data\coco\val2017`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

The script validates supported images, selects exactly 500 using seed `42`,
and copies them into `data\coco\subset_500`. Its `manifest.json` records the
source, seed, filenames, and SHA-256 hashes. Re-running verifies an existing
matching subset without overwriting it; a different nonempty output directory
is left untouched and reported as an error.

## Encode and decode

Provide a trained/exported SavedModel and an input image:

```powershell
.\.venv\Scripts\python.exe encode_image.py .\saved_models\my_experiment `
  --image .\test_im.png --save_dir .\out --secret Hello

.\.venv\Scripts\python.exe decode_image.py .\saved_models\my_experiment `
  --image .\out\test_im_hidden.png
```

The secret is padded and BCH encoded by the scripts and must be at most seven
UTF-8 characters. Encoding writes both the watermarked image and residual into
the output directory.

## Notes

- Training and exported inference models use TensorFlow 1.x graph semantics
  under TensorFlow 2.15.1; this is not a Keras 3 or eager-mode rewrite.
- The legacy detector is not included in this core workflow.
- Existing TensorFlow 1.x SavedModels and checkpoints have not been tested
  for compatibility with this port; a fresh model trained here is the
  supported path.
- The original project is MIT licensed. LPIPS retains its own license in
  `lpips/LICENSE`.
