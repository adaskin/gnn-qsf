
# Learnable Quantum Spectral Filters for Hybrid Graph Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2507.05640-b31b1b.svg)](https://arxiv.org/abs/2507.05640)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/mit)

Official implementation for the paper:  
**"Learnable Quantum Spectral Filters for Hybrid Graph Neural Networks"**  
*Ammar Daskin ([arXiv 2507.05640](https://arxiv.org/abs/2507.05640)), July 2025.*

## Overview
This repository contains PyTorch/PennyLane code for:
1. **Quantum Spectral Filters**: Parameterized quantum circuits that approximate graph Laplacian eigenspaces
   1. `parameterized_qft.py`
2. **Hybrid QGNNs**: Quantum-classical graph neural networks with exponential compression
   1. `qgnn.py`
3. **Geometric Graph Learning**: Reported figures and saved runs used in the paper on molecular and geometric datasets
   1. `saved_runs/` and `figures`

Key features:
- Quantum Fourier transform-based spectral filtering
- Graph structure-aware qubit connections
- Hybrid quantum-classical architecture
- TU dataset benchmarks (MUTAG, Letter-high, etc.)

## Paper Abstract
> In this paper, we describe a parameterized quantum circuit that can be considered as convolutional and pooling layers for graph neural networks. The circuit incorporates the parameterized  quantum Fourier circuit where the qubit connections for the controlled gates derived from the Laplacian operator.
Specifically, we show that the eigenspace of the Laplacian operator of a graph can be approximated by using QFT based circuit whose connections are determined from the adjacency matrix. For an $N\times N$ Laplacian, this approach yields an approximate polynomial-depth circuit requiring only $n=log(N)$ qubits. These types of circuits can eliminate the expensive classical computations for approximating the learnable functions of the Laplacian through Chebyshev polynomial or Taylor expansions.
 Using this circuit as a convolutional layer provides an $n-$ dimensional probability vector that can be considered as the filtered and compressed graph signal.  Therefore, the circuit along with the measurement can be considered a very efficient convolution plus pooling layer that transforms an $N$-dimensional signal input into $n-$dimensional signal with an exponential compression. 
We then apply a classical neural network prediction head  to the output of the circuit to construct a complete graph neural network. Since the circuit incorporates geometric structure through its graph connection-based approach, we present graph classification results for the benchmark datasets listed in TUDataset library  (AIDS, Letter-high, Letter-med, Letter-low, MUTAG, ENZYMES, PROTEINS, COX2, BZR, DHFR, MSRC-9). Using only [1-100] learnable parameters for the quantum circuit and minimal classical layers (1000-5000 parameters) in a generic setting, the obtained results are comparable to and in some cases better than many of the baseline results, particularly for the cases when geometric structure plays a significant role.

## Code Structure
- `/qgnn.py`: Main implementation for hybrid quantum-classical GNN
- `/parameterized_qft.py`: Quantum circuit for spectral filter approximation
- `requirements.txt`: Python dependencies

## Installation
```bash
git clone https://github.com/adaskin/gnn-qsf.git
cd gnn-qsf
pip install -r requirements.txt
```

## Usage
### Run QGNN experiments
```python
# Example: Letter-low dataset
python qgnn.py --dataset Letter-low --n_layers 1 --epochs 50
```

### Train spectral filters
```python
# 3-qubit circuit example
python parameterized_qft.py --n_qubits 3 --n_layers 5 --epochs 1000
```

## Citation
```bibtex
@article{daskin2025qspectral,
  title={Learnable Quantum Spectral Filters for Hybrid Graph Neural Networks},
  author={Daskin, Ammar},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
}
```

## License
See [LICENSE](LICENSE) for details.
