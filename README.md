# Pre-Generating Multi-Difficulty PDE Data For Few-Shot Neural PDE Solvers

Official code repository for the research paper on difficulty transfer in neural PDE solvers.

**Paper**: Pre-Generating Multi-Difficulty PDE Data For Few-Shot Neural PDE Solvers
**Authors**: Naman Choudhary*, Vedant Singh*, Ameet Talwalkar, Nicholas Matthew Boffi, Mikhail Khodak, Tanya Marwah
**Institution**: Machine Learning Department, Carnegie Mellon University
*Equal contribution

## Dataset

We provide pre-generated datasets for training neural PDE solvers with multi-difficulty data:

- **Hugging Face Dataset**: [PreGen-NavierStokes-2D](https://huggingface.co/datasets/sage-lab/PreGen-NavierStokes-2D)
- **Dataset Type**: Multi-difficulty 2D incompressible Navier-Stokes equations with varying:
  - **Geometry complexity** (number and placement of obstacles)
  - **Physics complexity** (Reynolds numbers)
  - **Combined difficulty variations**

## Overview

This research addresses a key bottleneck in neural PDE solvers: the computational cost of generating training data often exceeds the cost of training the model itself. We demonstrate that by strategically combining low and medium difficulty examples with high-difficulty examples, we can achieve **8.9× computational savings** on dataset pre-generation while maintaining the same error levels.

## Repository Structure

- **CNO_Experiments/**: Experiments with Convolutional Neural Operators (CNO)
  - Training scripts for different difficulty levels (L1-L5)
  - Fine-tuning and evaluation scripts
- **dataset_gen/**: Dataset generation utilities for creating multi-difficulty PDE data
- **Poseidon_mixing_Exp/**: Experiments with Poseidon-based mixing approaches
- **requirements.txt**: All project dependencies

## Installation

```bash
git clone https://github.com/Naman-Choudhary-AI-ML/Geo-UPSplus.git
cd Geo-UPSplus
pip install -r requirements.txt
```

### Docker

Alternatively, build and run the Docker container:

```bash
docker build -t geo-upsplus .
docker run -it geo-upsplus
```

## Quick Start

### Training a Model

```bash
cd CNO_Experiments
python TrainCNO_time_L.py  # Train on low difficulty data
python TrainCNO_time_L5.py # Train on highest difficulty data
```

### Testing

```bash
python TestCNO_ALL.py
```

## Key Findings

- **Difficulty Transfer**: Low and medium difficulty data transfers effectively to high-difficulty problems
- **8.9× Compute Savings**: Combining multi-difficulty data reduces classical solver compute by ~8.9× while maintaining error rates
- **Practical Impact**: Enables training on harder problems with significantly reduced computational overhead

## Citation

```bibtex
@article{choudhary2025pde,
  title={Pre-Generating Multi-Difficulty PDE Data For Few-Shot Neural PDE Solvers},
  author={Choudhary, Naman and Singh, Vedant and Talwalkar, Ameet and Boffi, Nicholas Matthew and Khodak, Mikhail and Marwah, Tanya},
  journal={arXiv preprint},
  year={2025}
}
```

## Project Website

For more details and visualizations: [https://naman-choudhary-ai-ml.github.io/pde-difficulty-transfer/](https://naman-choudhary-ai-ml.github.io/pde-difficulty-transfer/)

## Contact

For questions or issues, please contact: [namanchoud@andrew.cmu.edu](mailto:namanchoud@andrew.cmu.edu)

---

© 2025 Carnegie Mellon University
