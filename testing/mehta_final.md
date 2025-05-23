# Matrix Analysis and Low-Rank Approximation Methods

This repository contains a comprehensive collection of code for analyzing, visualizing, and comparing various low-rank matrix approximation methods with a focus on reinforcement learning techniques across different ranks.

## Project Overview

This project explores different approaches to matrix approximation, with particular emphasis on:

1. **Reinforcement Learning (RL) Approaches**: Implementation of novel RL-based approximation methods
2. **Rank Comparison**: Analysis of how different methods perform across varying ranks
3. **Ensemble Methods**: Combining multiple approximation techniques for improved results
4. **Visualization**: Comprehensive tools for visualizing results and performance metrics

## Key Files and Their Functions

### Core Implementations

- **cell1.py**: Core implementation of the RL-based column selection approach using Deep Q-Networks (DQN). Includes the environment definition, neural network architecture, and training procedures for the RL agent.

- **matrix_optimization.py**: Contains optimized algorithms for matrix approximation with various techniques including randomized methods and early stopping criteria.

- **matrix_analysis.py**: Tools for analyzing matrix properties and calculating error metrics like Frobenius and spectral norms.

- **matrix_experiments.py**: Framework for running controlled experiments across different matrix types and recording results.

### Analysis and Visualization

- **rank10_analysis.py**: Detailed analysis of the performance of various methods at rank 10, including comprehensive visualizations, error metrics, and efficiency comparisons.

- **visualize_optimization.py**: Creates visualizations to show the effect of different optimization strategies on approximation quality and performance.

- **visualize_early_stopping.py**: Analyzes and visualizes the impact of early stopping criteria on approximation algorithms.

### Specialized Tools

- **multi_matrix.py**: Implements techniques for handling multiple matrices simultaneously, including batch processing and parallel computation.

- **florida_downloader.py**: Utility for downloading test matrices from the University of Florida Sparse Matrix Collection.

- **check_800.py**: Validation script to verify the correctness of approximations, particularly for large matrices around size 800.

- **test_unoptimized_rsvd.py**: Benchmark testing for the unoptimized randomized SVD implementation as a baseline.

## Key Features and Contributions

1. **Novel RL Column Selection**: An innovative approach using reinforcement learning for intelligent column selection in matrix approximation.

2. **Comprehensive Comparisons**: Detailed analysis comparing traditional methods (SVD, randomized SVD) with the novel RL-based approaches.

3. **Performance Optimization**: Various techniques for improving computational efficiency including early stopping and specialized algorithms.

4. **Visualization Tools**: Rich set of visualization functions for exploring matrix properties, approximation errors, and comparative performance.

5. **Rank Analysis**: In-depth examination of how approximation quality and computation time vary with different target ranks.

## Results Highlights

- The RL-based column selection approach shows competitive accuracy compared to traditional methods with potential for better scaling.
  
- Randomized methods demonstrate significant speedups over deterministic SVD while maintaining acceptable error bounds.

- Ensemble methods that combine multiple approximation techniques show improved robustness across different matrix types.

- Different methods excel for different matrix types, with the best method often depending on the specific properties of the input.

## Usage

Various scripts can be run independently to test different aspects of matrix approximation:

```bash
# Run the RL-based approximation method
python cell1.py

# Analyze performance at rank 10
python rank10_analysis.py

# Test with optimization strategies
python matrix_optimization.py

# Visualize early stopping effects
python visualize_early_stopping.py
```

## Methodology

The project applies a multi-faceted approach to matrix approximation:

1. **Traditional Methods**: Deterministic SVD and randomized SVD as baselines
2. **RL Methods**: Using Deep Q-Networks to learn optimal column selection
3. **Hybrid Approaches**: Combining analytical methods with learned components
4. **Ensemble Techniques**: Averaging or selecting from multiple approximation methods

Each method is evaluated on accuracy (Frobenius and spectral error), computational efficiency, and scaling behavior with matrix size and target rank.