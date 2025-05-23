# Matrix Approximation using Reinforcement Learning

This repository contains code for our research on using Reinforcement Learning techniques for low-rank matrix approximation. Our approach outperforms state-of-the-art methods for column selection-based matrix approximation.

## Project Overview

We tackle the problem of low-rank matrix approximation, comparing traditional methods like SVD with reinforcement learning-based approaches for column selection. Our research demonstrates that RL agents can learn effective strategies for selecting representative columns to create low-rank approximations with competitive or superior error metrics compared to classical methods.

## File Structure

### Ensemble Code and Visualising Codes in Testing Folder

#### Core Implementations

- **cell1.py**: Core implementation of the RL-based column selection approach using Deep Q-Networks (DQN). Includes the environment definition, neural network architecture, and training procedures for the RL agent.

- **matrix_optimization.py**: Contains optimized algorithms for matrix approximation with various techniques including randomized methods and early stopping criteria.

- **matrix_analysis.py**: Tools for analyzing matrix properties and calculating error metrics like Frobenius and spectral norms.

- **matrix_experiments.py**: Framework for running controlled experiments across different matrix types and recording results.

#### Analysis and Visualization

- **rank10_analysis.py**: Detailed analysis of the performance of various methods at rank 10, including comprehensive visualizations, error metrics, and efficiency comparisons.

- **visualize_optimization.py**: Creates visualizations to show the effect of different optimization strategies on approximation quality and performance.

- **visualize_early_stopping.py**: Analyzes and visualizes the impact of early stopping criteria on approximation algorithms.

#### Specialized Tools

- **multi_matrix.py**: Implements techniques for handling multiple matrices simultaneously, including batch processing and parallel computation.

- **florida_downloader.py**: Utility for downloading test matrices from the University of Florida Sparse Matrix Collection.

- **check_800.py**: Validation script to verify the correctness of approximations, particularly for large matrices around size 800.

- **test_unoptimized_rsvd.py**: Benchmark testing for the unoptimized randomized SVD implementation as a baseline.


These codes are present in a seprate folder `testing`.

### Jupyter Notebooks

- `mid.ipynb` - Mid-evaluation code with simpler implementations of SVD and RL column selection methods.
- `final.ipynb` - Final comprehensive notebook containing our state-of-the-art implementations. This notebook is where we defeated the SOTA using RTX-3050 ti GPU. Here we were testing on different OSCIL-DCOP matrices and have shown to reduce time significantly.

### Results and Data

- `data/` - Contains matrix data files from the SuiteSparse Matrix Collection.
- `models/` - Saved trained RL agent models.
- `results/` - Experiment results, plots, and analysis which we generated.
- `proof/` - Additional analysis and proof-of-concept experiments.

## Key Methods Implemented

1. **Traditional Methods**
   - Deterministic SVD
   - Randomized SVD 
   - CUR Decomposition

2. **RL-based Methods**
   - DQN-based Column Selection
   - A2C-based Column Selection
   - Enhanced RL environments with different state representations
   - Ensemble RL-based Column Selection (our SOTA approach)

## Dependencies

```
numpy
scipy
torch
matplotlib
seaborn
pandas
tqdm
colorama
requests
```

## How to Run

### Setup

```bash
# Install dependencies
pip install numpy scipy torch matplotlib seaborn pandas tqdm colorama requests
```

### Running the Basic Example

```bash
# Run the basic DQN column selection example
python testing/cell1.py
```

### Running Matrix Experiments

```bash
# Run comprehensive matrix experiments
python testing/matrix_experiments.py
```

### Using Jupyter Notebooks

```bash
# Start Jupyter notebook server
jupyter notebook

# Open the final notebook
# Navigate to final.ipynb
```

## Using the Florida Matrix Collection

1. The `florida_downloader.py` script can be used to download matrices from the SuiteSparse Matrix Collection.
2. You can specify which matrices to download in the script.
3. Downloaded matrices are automatically stored in the `data/` directory.

## Main Results

Our best-performing method is the Ensemble RL-based Column Selection approach, which:

1. Combines multiple RL agents trained on different environments
2. Outperforms traditional SVD and CUR methods in approximation quality
3. Shows better generalization to unseen matrices
4. Offers an excellent trade-off between approximation quality and computational complexity

## Further Information

- `Final_report.pdf` - Contains the final report of our research, including detailed results and analysis.
- `presentation.pdf` - Final Presentation slides summarizing our research findings.
- `NA_MID_EVAL.pdf` - Mid-evaluation slides and results.
```