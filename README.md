# PROVIS: A Text-to-Image Diffusion Model

PROVIS (Progressive Vision-to-Image Synthesis) is a state-of-the-art text-to-image diffusion model designed for high-fidelity image synthesis from natural language prompts. It integrates a CLIP-based text encoder and a UNet-based denoising network, trained using a DDPM-style scheduler on 64x64 images.

---

## 🔧 Model Architecture

### Text Encoder
- **Backbone:** `CLIPTextModel`
- **Token Embedding:** `(49408, 512)`
- **Position Embedding:** `(77, 512)`
- **Transformer Depth:** Multiple `BasicTransformerBlock`s with self and cross-attention layers.

### UNet Backbone
- **Downsampling:** Hierarchical convolutional and transformer-based blocks.
- **Bottleneck:** `UNetMidBlock2DCrossAttn` with attention and feedforward layers.
- **Upsampling:** Multi-stage `UpBlock2D` and `CrossAttnUpBlock2D` with resnet-style residual blocks.

### Key Modules
- `ResnetBlock2D`
- `Attention` (self and cross-attention)
- `GEGLU` feedforward networks
- `GroupNorm`, `LayerNorm`, and `SiLU` activations

### Parameters
- **Total Parameters:** `1,063,773,419`
- **Trainable Parameters:** `1,063,773,419`

---

## 🧠 Training Setup
- **Scheduler:** Denoising Diffusion Probabilistic Model (DDPM)
- **Image Size:** 64x64
- **Loss Function:** Mean Squared Error (MSE) between predicted and target noise
- **Tokenization:** CLIP Tokenizer

---

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/PROVIS.git
cd PROVIS

# Install dependencies
pip install -r requirements.txt
```

### Inference Example
```python
from provis import PROVISModel
model = PROVISModel.from_pretrained('path/to/checkpoint')
image = model.generate("a fantasy landscape with glowing trees")
image.save("output.png")
```

---

## 🧩 Features
- CLIP-powered text conditioning
- Transformer-augmented UNet backbone
- Multi-scale attention and residual learning
- High-capacity architecture (1B+ parameters)

---

## 📁 Directory Structure
```
PROVIS/
├── provis/
│   ├── model.py
│   ├── layers/
│   ├── tokenizer/
│   └── scheduler.py
├── scripts/
│   ├── train.py
│   └── inference.py
├── configs/
├── checkpoints/
└── README.md
```

---

## 🧪 Evaluation
- Coming soon: FID, IS, and human preference benchmarks on MS-COCO and LAION datasets.

---

## 📝 License
[MIT License](LICENSE)

---

## 🙌 Acknowledgements
This project builds on open-source contributions from OpenAI's CLIP, Hugging Face Diffusers, and the academic community working on diffusion models.

---

## 🚀 Contributing
Pull requests and feature suggestions are welcome! Please check out the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📫 Contact
For questions or collaborations, reach out via GitHub issues or email at `your.email@example.com`

