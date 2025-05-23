# Matrix Approximation Methods Analysis at Rank 10

Date: May 8, 2025

## Key Findings

- **Best Accuracy Method**:     Deterministic SVD
Name: 0, dtype: object
- **Fastest Method**:     CUR
Name: 4, dtype: object
- **Most Efficient Method**:     Deterministic SVD
Name: 0, dtype: object
- **Best Overall Method**: Deterministic SVD

## Randomized SVD Performance

- **Speedup vs. Deterministic SVD**: 21.99x
- **Error Ratio to Deterministic SVD**: 1.0000x

## Error Performance

| Method | Mean Frobenius Error | Mean Spectral Error |
|--------|----------------------|---------------------|
| Deterministic SVD | 0.1843 | 0.1499 |
| Randomized SVD | 0.1843 | 0.1499 |
| RL-DQN | 0.9966 | 1.0000 |
| RL-A2C | 0.9597 | 0.9999 |
| CUR | 2.6888 | 4.2360 |

## Time Performance

| Method | Mean Time (seconds) |
|--------|---------------------|
| Deterministic SVD | 2.3727 |
| Randomized SVD | 0.1079 |
| RL-DQN | 0.2279 |
| RL-A2C | 2.5949 |
| CUR | 0.0804 |

## Matrix-Specific Insights

| Matrix | Best Method | Fastest Method |
|--------|-------------|---------------|
| oscil_dcop_10 | Deterministic SVD | CUR |
| oscil_dcop_11 | Deterministic SVD | Randomized SVD |
| oscil_dcop_12 | Deterministic SVD | CUR |
| oscil_dcop_13 | Deterministic SVD | CUR |

## Recommendations

- For highest accuracy: Use     Deterministic SVD
Name: 0, dtype: object
- For fastest computation: Use     CUR
Name: 4, dtype: object
- For best efficiency (error/time ratio): Use     Deterministic SVD
Name: 0, dtype: object
- Best overall balance: Use Deterministic SVD
