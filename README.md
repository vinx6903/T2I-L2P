<div align="center">

# L2P: Unlocking Latent Potential for Pixel Generation

<p>
  <a href="https://nju-pcalab.github.io/projects/L2P/">
    <img src="https://img.shields.io/badge/Project-Page-1f72ff?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2605.12013">
    <img src="https://img.shields.io/badge/arXiv-2605.12013-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
</p>

<p>
  <em>An efficient transfer paradigm enabling high-quality, end-to-end pixel-space diffusion with minimal computational overhead and data requirements.</em>
</p>

<div align="center">
<sub>⭐ If L2P helps your research or product, please consider giving the repo a star ⭐</sub>
</div>


</div>

---

## 📰 News

- **\[2026/05\]** Technical report released.

---

## 🗺️ Roadmap

| Status | Item |
| :---: | :--- |
| 🛠️ | 1K inference code & weights |
| 🛠️ | Training code |
| 🛠️ | 4K/8K/10K UHR generation |
| 🛠️ | Compatibility with more LDM T2I model |

---

## 📦 Installation

<!-- ```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/L2P.git
cd L2P

# 2. (Recommended) Create a clean Python environment
conda create -n l2p python=3.10 -y
conda activate l2p

# 3. Install the package in editable mode (reads dependencies from pyproject.toml)
pip install -e .
``` -->


---

<!--
## 🗂️ Pretrained Weights

Download the following assets and place them under `ckpt/` and `models/` (paths can be customized in the scripts):

| Component | Source |
| :--- | :--- |
| L2P main DiT (1K) | *(release link coming soon)* |
| Z-Image text encoder | [`Z-Image-Turbo/text_encoder`](https://huggingface.co/) |
| Z-Image tokenizer | [`Z-Image-Turbo/tokenizer`](https://huggingface.co/) |
| Pixel-space initialization | `Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors` |

Then update the paths inside `inference.py` / `train_run.sh` accordingly.

---
-->

## 🎨 Inference
```python
import torch
from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig

main_model_path = "/path/model-1k-merge.safetensors"

text_encoder_paths = [
    "/path/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors",
]

tokenizer_path = "/path/Z-Image-Turbo/tokenizer"

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[main_model_path]),
        ModelConfig(path=text_encoder_paths),
    ],
    tokenizer_config=ModelConfig(path=tokenizer_path),
)

prompt = "an origami pig on fire in the middle of a dark room with a pentagram on the floor"

image = pipe(
    prompt=prompt,
    seed=42,
    rand_device="cuda",
    num_inference_steps=30,
    cfg_scale=2.0,
    height=1024,
    width=1024,
)

image.save("example.png")
```

### Gradio Demo

Launch a multi-GPU web UI:

```bash
python app.py
```

The demo auto-detects free GPUs, dispatches each request to an idle device, and exposes a Gradio interface at `http://0.0.0.0:23231`.

---

## 🏋️ Training

The full training pipeline consists of four steps:
**(1)** prepare the Z-Image base weights → **(2)** convert them into a pixel-space initialization → **(3)** launch training → **(4)** merge the trained delta back with the pixel-init weights for inference.

### Step 1 · Prepare Z-Image weights

Download the official **Z-Image-Turbo** checkpoint from Hugging Face:

- 🤗 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)



### Step 2 · Offline weight conversion (latent → pixel init)

Convert the latent-space DiT weights into a **pixel-space initialization** that L2P can fine-tune from:

```bash
python examples/z_image/L2P_convert_weight.py \
  --latent_ckpt_files \
    /path/to/Z-Image-Turbo/transformer/diffusion_pytorch_model-00001-of-00003.safetensors \
    /path/to/Z-Image-Turbo/transformer/diffusion_pytorch_model-00002-of-00003.safetensors \
    /path/to/Z-Image-Turbo/transformer/diffusion_pytorch_model-00003-of-00003.safetensors \
  --output_path ./pretrain_weight/Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors
```

### Step 3 · Launch training

**Standard training** :

```bash
bash train_run.sh
```

**Low-VRAM training** (single GPU < 24 GB VRAM):

```bash
bash train_run_low_VRAM.sh
```

#### Dataset format

Provide a directory of images plus a CSV metadata file:

```
data/
├── images/                # raw image folder
└── metadata.csv           # columns: file_name, text, ...
```

### Step 4 · Offline weight merge (for inference)

```bash
python merge_weights.py \
  --file_a ./models/train/L2P_Standard/step-xxx.safetensors \
  --file_b ./pretrain_weight/Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors \
  --file_out ./models/train/L2P_Standard/model-merge.safetensors
```

- `--file_a`: trained checkpoint from Step 3
- `--file_b`: pixel-init weights from Step 2
- `--file_out`: merged single-file weight
---

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{chen2026l2p,
  title   = {L2P: Unlocking Latent Potential for Pixel Generation},
  author  = {Chen, Zhennan and Zhu, Junwei and Chen, Xu and Zhang, Jiangning and
             Chen, Jiawei and Zeng, Zhuoqi and Zhang, Wei and Wang, Chengjie and
             Yang, Jian and Tai, Ying},
  journal = {arXiv preprint arXiv:2605.12013},
  year    = {2026}
}

@article{chen2025dip,
  title   = {DiP: Taming Diffusion Models in Pixel Space},
  author  = {Chen, Zhennan and Zhu, Junwei and Chen, Xu and Zhang, Jiangning and
             Hu, Xiaobin and Zhao, Hanzhen and Wang, Chengjie and Yang, Jian and
             Tai, Ying},
  journal = {arXiv preprint arXiv:2511.18822},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

L2P is built upon the excellent open-source work of
[**DiffSynth-Studio**](https://github.com/modelscope/DiffSynth-Studio),
[**Z-Image**](https://github.com/Tongyi-MAI/Z-Image).

