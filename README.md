# latent-space-xgboost

This repository demonstrates how XGBoost can be used to predict latent variables that represent very high-dimensional multivariate timeseries. Most XGBoost tutorials focus on the problem of multi-class classification or scalar regression. Tree-based algorithms for prediction of a high-dimensional vector can become intractable, therefore in this repository, we focus on the problem of enabling vector regression with XGBoost. We use an autoencoder to first perform dimensionality reduction of multivariate timeseries to obtain the corresponding latent variables from a compact latent space by extracting salient temporal features. Subsequently, we train XGBoost models to create a mapping from the input space to the latent space. This demo uses simulated dataset and field dataset from unconventional reservoirs, where the input is a set of well properties and the output is the latent variables representing multiphase production profiles.

<img src="/readme/workflow.png" width="600">

```
latent-space-xgboost
├── sim-case
│   ├── trees-scalar.ipynb
│   ├── trees-vector-independent.ipynb
│   ├── trees-vector-independent-nans.ipynb
│   ├── trees-vector-independent-impute.ipynb
│   ├── trees-vector-dependent.ipynb
│   ├── trees-vector-dependent-nans.ipynb
│   └── trees-vector-dependent-impute.ipynb
├── field-case
│   ├── fcnn-vector-case-1.ipynb
│   ├── fcnn-vector-case-2.ipynb
│   ├── trees-vector-independent.ipynb
│   ├── trees-vector-independent-nans.ipynb
│   ├── trees-vector-dependent.ipynb
│   └── trees-vector-dependent-nans.ipynb
```

## Workflow 

For latent variables vector regression with XGBoost, the prediction of the latent variables can be done independently (i.e., individual XGBoost model for each variable in the latent vector) or dependently (i.e., hierarchical XGBoost models that use any predicted variable as input for the next model). 

![Methods](/readme/methods.png)

The notebooks in this repository have been labeled accordingly and you can find more description inside. For the simulated dataset, when some of the input data has missing features, a higher error is observed for both dependent and independent prediction methods. A relatively lower error is observed when we let XGBoost handle the NaNs implicitly, rather than performing a global mean imputation. 

![Sim](/readme/sim-compare.png)

For the field dataset, dependent and independent prediction methods yield similar results. A relatively higher error is also observed with global mean imputation. 

![Field](/readme/field-compare.png)

Interestingly, with this rather noisy dataset, a fully-connected neural network (FCNN, in lieu of XGBoost) seems to outperform XGBoost even when the missing features are imputed, as FCNN has no mechanism to handle missing values. 

