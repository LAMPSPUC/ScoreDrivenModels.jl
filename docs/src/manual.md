# Manual

## Model Specification

```@docs
Model
```

## Optimization Algorithms

ScoreDrivenModels.jl allows users to tests different optimization methods, in particular
it has a common interface to easily incorporate algorithms available on [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)

All optimization methods can receive the following keyword arguments
    * `f_tol` - Relative tolerance in changes of the objective value. Default is `1e-6`.
    * `g_tol` - Absolute tolerance in the gradient, in infinity norm. Default is `1e-6`.
    * `iterations` - How many iterations will run before the algorithm gives up? Default is `10^5`.
    * `LB` - Lower bound of the initial points. Default is `0.0`.
    * `UB` - Upper bound of the initial points. Default is `0.6`.

The [ScoreDrivenModels.IPNewton](@ref) allows users to perform box-constrained optimization, 
more details in [ScoreDrivenModels.IPNewton](@ref).

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
and then one can recover the original parameter by ``f = \left(h\right)^-1(\tilde f)``. 
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

ScoreDrivenModels.jl allows users to generate, a point forecast, confidence interval 
forecasts or an ensemble of scenarios. Point forecasts are obtained using the function `forecast` 
and the ensemble of scenarios are obtained using the function `simulate`

```@docs
forecast
forecast_ci
simulate
```

## ScoreDrivenModels distributions

The following section presents how every distribution is parametrized, its score, fisher information
and the `time_varying_params` map. Every distribution is originally imported to ScoreDrivenModels.jl
from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).

```@docs
ScoreDrivenModels.Poisson
ScoreDrivenModels.Normal
ScoreDrivenModels.LogNormal
ScoreDrivenModels.Beta
ScoreDrivenModels.Gamma
ScoreDrivenModels.Weibull
```

## Implementing a new distribution

If you want to add a new distribution please feel free to make a pull request.

Each distribution must have the following methods:
* `score`
* `fisher information`
* `log likelihood`
* `link interface`
    * `link`
    * `unlink`
    * `jacobian_link`
* `update_dist`
* `num_params`

The details of the new distribution must be documented following the example in
[`Normal`](@ref) and added to the [ScoreDrivenModels distributions](@ref) section.

# Reference

```@docs
ScoreDrivenModels.Unknowns
```