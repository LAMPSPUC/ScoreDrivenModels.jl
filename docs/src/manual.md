# Manual

For a concise overview of the package, please read the [package paper](https://arxiv.org/abs/2008.05506). 
It provides a high-level presentation of the theory behind score-driven models and showcases the features of the package as well as examples.

## Model Specification

```@docs
ScoreDrivenModel
```

## Optimization Algorithms

ScoreDrivenModels.jl allows users to use different optimization methods, in particular
it has a common interface to easily incorporate algorithms available on [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)

All optimization methods can receive the following keyword arguments
* `f_tol` - Relative tolerance in changes of the objective value. Default is `1e-6`.
* `g_tol` - Absolute tolerance in the gradient, in infinity norm. Default is `1e-6`.
* `iterations` - Maximum number of iterations. Default is `10^5`.
* `LB` - Lower bound of the initial points. Default is `0.0`.
* `UB` - Upper bound of the initial points. Default is `0.6`.

[`ScoreDrivenModels.IPNewton`](@ref) allows users to perform box-constrained optimization.

```@docs
ScoreDrivenModels.NelderMead
ScoreDrivenModels.LBFGS
ScoreDrivenModels.IPNewton
```

## Recursion

## Links

Links are reparametrizations utilized to ensure certain parameter is within its original domain. 
For instance, for a particular distribution, one might want to ensure that the time varying 
parameter is positive: ``f \in \mathbb{R}^+``. The way to do this is to model ``\tilde{f} = \ln{f}``. 
More generally, one can establish that ``\tilde{f} = h(f)``. We refer to this procedure as 
**linking**. When a parameter is linked, the GAS recursion happens in the domain of ``\tilde{f}`` 
and then one can recover the original parameter by ``f = \left(h\right)^{-1}(\tilde f)``. 
We refer to this procedure as **unlinking**. The new recursion becomes:

```math
\begin{equation*}\left\{\begin{array}{ccl}
    f_{t} &=& h^{-1}(\tilde f_t), \\
    \tilde f_{t+1} &=& \omega + \sum_{i=1}^p A_{i}\tilde s_{t-i+1} + \sum_{j=1}^q B_{j}\tilde f_{t-j+1}
    \end{array}
    \right.
\end{equation*}
```

Notice that a different parametrization alters the dynamics of the model. For example, 
the GAS(1,1) model with Normal distribution and scaling ``d = 1`` is equivalent to the well-known 
GARCH(1, 1) model. Conversely, if a different parametrization is utilized, the model will 
no longer be equivalent.

### Types of links

The abstract type `Link` subsumes any type of link that can be expressed.

```@docs
ScoreDrivenModels.IdentityLink
ScoreDrivenModels.LogLink
ScoreDrivenModels.LogitLink
```

### Link functions

```@docs
ScoreDrivenModels.link
ScoreDrivenModels.unlink
ScoreDrivenModels.jacobian_link
```

## Forecasting

ScoreDrivenModels.jl allows users to generate point forecasts, confidence intervals 
forecasts or ensembles of scenarios. Point forecasts are obtained using the function `forecast` 
and ensembles of scenarios are obtained using the function `simulate`.

```@docs
forecast_quantiles
simulate
```

## ScoreDrivenModels distributions

The following section presents how every distribution is parametrized, its score, Fisher information
and the `time_varying_params` map. Every distribution is originally imported to ScoreDrivenModels.jl
from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). Some distributions may have different
parametrizations from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl),
this is handled internally.

| Distribution | Identity scaling | Inverse and inverse square root scalings |
| :------- |:---------------:|:--------------------:|
|[`Beta`](@ref)| ✓ | ✓  |
|[`BetaFourParameters`](@ref)| ✓ |  x  |
|[`Exponential`](@ref)| ✓ | ✓  |
|[`Gamma`](@ref)| ✓ | ✓  |
|[`LogitNormal`](@ref)| ✓ | ✓  |
|[`LogNormal`](@ref)| ✓ | ✓  |
|[`NegativeBinomial`](@ref)| ✓ |  x  |
|[`Normal`](@ref)| ✓ | ✓  |
|[`Poisson`](@ref)| ✓ | ✓  |
|[`TDist`](@ref)| ✓ | ✓  |
|[`TDistLocationScale`](@ref)| ✓ | ✓  |
|[`Weibull`](@ref)| ✓ |  x  |


```@docs
ScoreDrivenModels.Beta
ScoreDrivenModels.BetaFourParameters
ScoreDrivenModels.Exponential
ScoreDrivenModels.Gamma
ScoreDrivenModels.LogitNormal
ScoreDrivenModels.LogNormal
ScoreDrivenModels.NegativeBinomial
ScoreDrivenModels.Normal
ScoreDrivenModels.Poisson
ScoreDrivenModels.TDist
ScoreDrivenModels.TDistLocationScale
ScoreDrivenModels.Weibull
```

## Implementing a new distribution

If you want to add a new distribution please feel free to make a pull request.

Each distribution must have the following methods:
* [`ScoreDrivenModels.score!`](@ref)
* [`ScoreDrivenModels.fisher_information!`](@ref)
* [`ScoreDrivenModels.log_likelihood`](@ref)
* link interface
    * [`ScoreDrivenModels.link!`](@ref)
    * [`ScoreDrivenModels.unlink!`](@ref)
    * [`ScoreDrivenModels.jacobian_link!`](@ref)
* [`ScoreDrivenModels.update_dist`](@ref)
* [`ScoreDrivenModels.params_sdm`](@ref)
* [`ScoreDrivenModels.num_params`](@ref)

The details of the new distribution must be documented following the example in
[`Normal`](@ref) and added to the [ScoreDrivenModels distributions](@ref) section.
The new implemented distribution must also be added to the constant `DISTS` and exported in the
`distributions/common_interface.jl` file.

```@docs
ScoreDrivenModels.score!
ScoreDrivenModels.fisher_information!
ScoreDrivenModels.log_likelihood
ScoreDrivenModels.link!
ScoreDrivenModels.unlink!
ScoreDrivenModels.jacobian_link!
ScoreDrivenModels.update_dist
ScoreDrivenModels.params_sdm
ScoreDrivenModels.num_params
```

# Reference

```@docs
ScoreDrivenModels.Unknowns
```
