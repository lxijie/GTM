GTM: GENERAL TIME-SERIES MODEL FOR UNIVERSAL KNOWLEDGE REPRESENTATION OF MULTIVARIATE TIME-SERIES DATA

Universal knowledge representation is a central problem for multivariate time series(MTS) foundation
models and yet remains open. This paper investigates this problem from the first principle and it makes
four folds of contributions. First, a new empirical finding is revealed: time series with different time
granularities (or corresponding frequency resolutions) exhibit distinct joint distributions in frequency
domain. This implies a crucial aspect for learning universal knowledge, one that has been overlooked
by previous studies. Second, a novel Fourier knowledge attention mechanism is proposed to enable
learning time granularity-aware representations from both the temporal and frequency domains.
Third, an autoregressive blank infilling pre-training framework is incorporated to time series analysis
for the first time, leading to a generative tasks agnostic pre-training strategy. To this end, a unified
MTS foundation model is built, addressing the limitation that contemporary time series models
often require token, pre-training, or model-level customizations for downstream tasks adaption.
Fourth, extensive experiments show that GTM outperforms state-of-the-art (SOTA) methods across
all generative tasks, including long-term forecasting, anomaly detection, and imputation. We will
publicly release the code and models in the camera-ready version.

The pre trained model can be obtained through the following address: (https://huggingface.co/liangxj/GTM)
