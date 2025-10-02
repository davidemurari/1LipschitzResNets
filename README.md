# 1-Lipschitz ResNets — Reference Implementation

Minimal PyTorch code accompanying the paper **“Approximation theory for 1-Lipschitz ResNets”** (NeurIPS 2025, Poster). The repo contains small training/experiment scripts illustrating the architectures and constraints discussed in the paper.

## What’s here

- `main.py` — minimal training script for a toy setup / classification baseline.
- `mainMNIST.py` — minimal MNIST example.
- `experiments.sh`, `experimentsMNIST.sh` — example command sweeps to reproduce simple runs.

> Note: this is a lean research codebase meant to mirror the theory with the fewest moving parts.

## Setup

1. **Clone**
   ```bash
   git clone https://github.com/davidemurari/1LipschitzResNets
   cd 1LipschitzResNets
   ```

2. **Environment**
   - Python ≥ 3.9
   - PyTorch (CUDA optional)
   - Common Python stack: `numpy`, `torchvision`, `tqdm`, `matplotlib` (as needed)

   Example (CPU):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install numpy tqdm matplotlib
   ```

## Quick start

### A) Minimal run
```bash
python main.py
```

### B) MNIST example
```bash
python mainMNIST.py
```

### C) Example sweeps
Use the provided shell scripts (you can open and tweak hyper-parameters):
```bash
bash experiments.sh
bash experimentsMNIST.sh
```

## Notes

- The models implement residual blocks designed to be **1-Lipschitz** under simple norm constraints, matching the constructions studied in the paper (explicit-Euler steps of negative gradient flows; optional constrained linear interlayers). For full statements and proofs, please see the paper.
- Datasets are downloaded automatically by `torchvision` when used.

## Reference

If you use this code or build on the theory, please cite:

- Paper: **Approximation theory for 1-Lipschitz ResNets**. arXiv:2505.12003. DOI: 10.48550/arXiv.2505.12003.

### BibTeX
```bibtex
@article{murari2025approximation,
  title   = {Approximation theory for 1-Lipschitz ResNets},
  author  = {Murari, Davide and Furuya, Takashi and Sch{"o}nlieb, Carola-Bibiane},
  journal = {arXiv preprint arXiv:2505.12003},
  year    = {2025},
  doi     = {10.48550/arXiv.2505.12003}
}
```
