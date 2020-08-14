# ScoreDrivenModels.jl

| **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lampspuc.github.io/ScoreDrivenModels.jl/latest/)

[build-img]: https://travis-ci.org/LAMPSPUC/ScoreDrivenModels.jl.svg?branch=master
[build-url]: https://travis-ci.org/LAMPSPUC/ScoreDrivenModels.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/ScoreDrivenModels.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/ScoreDrivenModels.jl?branch=master

ScoreDrivenModels.jl is a Julia package for modeling, forecasting, and simulating time series with score-driven models, also known as generalized autoregressive score models (GAS). Implementations are based on the paper [Generalized Autoregressive Models with Applications](http://dx.doi.org/10.1002/jae.1279) by D. Creal, S. J. Koopman, and A. Lucas.

## Installation

This package is registered so you can simply `add` it using Julia's `Pkg` manager:
```julia
pkg> add StateSpaceModels
```

## Citing the package

If you use ScoreDrivenModels.jl in your work, we kindly ask you to cite the package [paper](https://arxiv.org/abs/2008.05506):

    @misc{bodin2020scoredrivenmodelsjl,
    title={ScoreDrivenModels.jl: a Julia Package for Generalized Autoregressive Score Models},
    author={Guilherme Bodin and Raphael Saavedra and Cristiano Fernandes and Alexandre Street},
    year={2020},
    eprint={2008.05506}
    }
