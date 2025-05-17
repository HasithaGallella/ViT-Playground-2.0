# ViT-Playground-2.0

**ViT-Playground-2.0** is a personal and practical environment designed for training, fine-tuning, LoRA-PEFT, testing, and experimenting with Vision Transformers (ViTs) and other large Transformer backbones. Built on PyTorch and CUDA, this repository supports workflows ranging from large-scale A100 multi-GPU cluster training to lightweight inference on personal CPU only devices.

---

## ✨ Features

- 🧠 **Train and Fine-tune Vision Transformers**  
  Includes support for standard supervised learning, unsupervised masking methods and lightweight fine-tuning methods like LoRA.

- 🔧 **Modular Architecture**  
  Easily extend or modify transformer components using subclass-based design.

- ⚙️ **Multi-Device Compatibility**  
  - ✅ Scalable training on SLURM-based HPC environments (e.g., worked on 640GB 8× A100 GPU cluster)
  - ✅ Fine-tuning on mid-range GPUs (e.g., worked on 6GB RTX 2060 Windows PC)  
  - ✅ Inference and low level module testing on CPU-only machines, including macOS (Tested on M4 MacBook)

- 🚀 **Built with PyTorch**  
  It utilizes native, from-scratch PyTorch modules and CUDA acceleration when available, with many test example implementations, such as How Backpropagation Works for transformer parameter updates at the low level.

---

## 📁 Project Structure

ViT-Playground-2.0/
├── src/                 # Core model architecture and utilities
├── train.py             # Entry point for local training
├── main_distributed.py  # Entry point for multi-GPU (e.g., Slurm) training
├── inference.py         # Inference/testing utilities
├── configs/             # YAML config files for various experiments
└── data/                # Dataset loading and preparation

---

## 🧪 Use Cases

This repo serves as my **testbed for Vision Transformer backbones**. It is actively used to:

- Evaluate different backbone variants under controlled settings
- Compare training behaviors across devices and scales
- Prototype new transformer modules or fine-tuning strategies

---

## 🔧 Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA (for GPU acceleration)

You can install the dependencies via:

```bash
pip install -r requirements.txt
```

---

📌 Notes
	•	This repository is actively evolving.
	•	It is tailored for my own research and experimentation but kept general enough for others to adapt.

---

🖥️ Example Usage

Train locally:

```bash
python train.py --fname configs/vit_base.yaml
```

Train on multiple GPUs:

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_distributed.py --fname configs/vit_large.yaml
```

---

🧑‍💻 Author

Maintained by @HasithaGallella.

If you find this useful or have suggestions, feel free to open an issue or a pull request.

---

📜 License

This project is released under the MIT License.

Let me know if you have any sudgestions to include to make this a more practicle codebase to explore Deep learning framework!
