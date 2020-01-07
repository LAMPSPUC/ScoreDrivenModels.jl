# GAS.jl

| **Build Status** | **Coverage** | **Documentation** |
|:-----------------:|:-----------------:|:-----------------:|
| [![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url] |[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://lampspuc.github.io/GAS.jl/latest/)

[build-img]: https://travis-ci.org/LAMPSPUC/GAS.jl.svg?branch=master
[build-url]: https://travis-ci.org/LAMPSPUC/GAS.jl

[codecov-img]: https://codecov.io/gh/LAMPSPUC/GAS.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/gh/LAMPSPUC/GAS.jl?branch=master

GAS.jl is a Julia package for modeling, forecasting and simulating time series with score-driven models, also known as dynamic conditional score models (DCS) or generalized autoregressive score models (GAS). Implementations are based on the paper [Generalized Autoregressive Models with Applications](http://dx.doi.org/10.1002/jae.1279) by D. Creal, S. J. Koopman and A. Lucas.

# Features
* Autoregressive structure
* Maximum likelihood estimation
* Monte Carlo simulation
* General link/unlink interface
* Available distributions:
  * Normal
  * Lognormal
  * Poisson
  * Beta
  * Gamma

# Roadmap
* Weibull distribution
* Student's t distribution
* Dynamic components structure