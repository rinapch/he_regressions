# 🔒 Effect of normalization on encrypted regressions

This repository contains experiments comparing traditional regression models with their homomorphically encrypted counterparts, analyzing both performance and prediction metrics. More specifically, we are looking at the effect of normalization on the performance of encrypted regressions. 

The motivation for this is the idea that normaliztion can reduce the length of the ciphertext, which can improve performance both in terms of time and quiality.

## Overview

We evaluate the trade-offs between standard machine learning approaches and privacy-preserving computations using homomorphic encryption (HE). The experiments focus on:
- Linear Regression (encrypted vs unencrypted)
- Logistic Regression (encrypted vs unencrypted)

We use financial data (FAANG stocks) for the first one, and medical data (release from the hospital prediction) for the second one.


### Quality Metrics
- Mean Squared Error (MSE) for Linear Regression
- Accuracy for Logistic Regression
- Execution Time for both

## Key Findings

- Normaliztion does improve both the metrics and the speed in both cases. 
- Z-score normalization is better than min-max normalization in terms of MSE and execution time.
- The encrypted model remains significantly slower than the unencrypted one
- The encrypted model is very suseptible to the choice of parameters for homomorphic encryption (especially the global scale). If not configured appropriately, the model will not run. This issue is even more pronounced in the linear regression case. For linear regression, we had to sacrifise precision to make the model function.


## Results
Speed is average speed of 5 epochs (in seconds).
Metrics are the best results obtained during training.


### No normalization:

| Model    | Metric | Speed |
| -------- | ------- | ------- |
| LogReg  |  0.66   | 0.1 |
| EncLogReg |  0.58  | 135 |
| LinReg    | 2.43    | 0.1|
| EncLinReg | 2232.43   | 2.92 | 


### Z-score normalization:

| Model    | Metric | Speed |
| -------- | ------- | ------- |
| LogReg  |  0.65   | 0.1 |
| EncLogReg |  0.68  | 133 |
| LinReg    | 0.73    | 0.1|
| EncLinReg | 2280.2146  | 2.95 | 


### Min-max normalization:
| Model    | Metric | Speed |
| -------- | ------- |
| LogReg  | 0.60 | 0.1 |
| EncLogReg | 0.60 | 133 |
| LinReg    | 0.77 | 0.1|
| EncLinReg | 1.50 | 2.89 |

