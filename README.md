---
tags:
- image-classification
- anti-spoofing
- deepfake-detection
- ai-generated-detection
- pytorch
license: mit
---

# Image Anti-Spoofing System

Detects whether an image is **real** or **fake** (AI-generated OR deepfake/manipulated).

Third project in a multi-modal anti-spoofing portfolio, alongside [Voice-Anti-Spoofing](https://huggingface.co/LaabhGupta/voice-antispoofing) and [Text-Anti-Spoofing](https://huggingface.co/LaabhGupta/Text-Anti-Spoofing).

## Architectures

Three architectures were trained and compared on the same data:

| Architecture | Description |
|---|---|
| `BaselineCNN` | Shallow 3-block Conv2D CNN |
| `DeeperCNN` | Deeper 4-block CNN with BatchNorm |
| `ViTModel` | ViT-B/16 (ImageNet-pretrained) adapted for RGB images via transfer learning |

## Training data

[`prithivMLmods/AI-vs-Deepfake-vs-Real`](https://huggingface.co/datasets/prithivMLmods/AI-vs-Deepfake-vs-Real) — original 3-way labels (`Artificial`, `Deepfake`, `Real`) merged into a binary task: `Real` (0) vs `Fake` (1, covering both AI-generated and deepfake images), to detect AI manipulation of any kind in one model.

## Results (test set accuracy)

| Model | Test Accuracy | File size |
|---|---|---|
| Baseline CNN | 98.80% | ~26MB |
| Deeper CNN | 99.20% | ~26MB |
| ViT | **100.00%** | ~343MB |

**Important caveat on the ViT's 100% score:** a perfect test-set score is a signal to scrutinize, not just celebrate. It likely reflects that this dataset's fake images (from a specific, fixed set of generators) have consistent, learnable artifacts, rather than the model having solved "detect any AI-manipulated image" universally. Performance against newer/unseen generators (e.g. Midjourney v6, Flux, Stable Diffusion 3, or generators not represented in this training set) is untested and likely to be lower - this is a common "concept drift" issue in this field, since fake-image generators keep improving and older detectors don't automatically generalize to them.

**Recommended checkpoint for deployment: `deeper_cnn_model.pth`.** At 99.20% accuracy and ~26MB (vs. ViT's 343MB), it offers a strong practical tradeoff between accuracy and deployment cost (faster cold starts, less bandwidth) with a more plausible, less potentially-overfit accuracy figure than the ViT's perfect score.

## Files in this repo

- `baseline_cnn_model.pth`
- `deeper_cnn_model.pth` (recommended)
- `vit_model.pth`
- `model.py` - architecture class definitions + preprocessing pipeline, required to load any of the above

## Input format

- Any image `PIL` can open (`.jpg`, `.png`, etc.)
- Resized to 224x224
- Normalized with standard ImageNet mean/std

`model.py` includes a `preprocess_image()` function that handles all of this automatically.

## Usage

```python
from huggingface_hub import hf_hub_download
import sys

model_py_path = hf_hub_download(repo_id="LaabhGupta/image-antispoofing", filename="model.py")
weights_path = hf_hub_download(repo_id="LaabhGupta/image-antispoofing", filename="deeper_cnn_model.pth")

sys.path.insert(0, model_py_path.rsplit("/", 1)[0])
from model import load_model, predict

model = load_model("deeper", weights_path, device="cpu")
label, confidence = predict("path/to/image.jpg", model, device="cpu")
print(label, confidence)
```
