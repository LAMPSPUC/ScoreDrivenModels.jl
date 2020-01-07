# Manual

## Model Specification

## Recursion

## Links

Links are reparametrizations utilized to ensure certain parameter is within its original domain, i.e. in a distribution one would like to ensure that the time varying parameter ``f \in \mathbb{R}^+``. The way to do this is to model ``\tilde{f} = \ln{f}``. More generally one can stablish that ``\tilde{f} = h(f)``. We refer to this procedure as **linking**. When the parameter is linked the GAS recursion happens in the domain of ``\tilde{f}`` and then one can recover the orginal parameter by ``f = \left(h\right)^-1(\tilde f)``. We refer to this procedure as **unlinking**. The new GAS recursion becomes.

```math
\begin{equation*}\left\{\begin{array}{ccl}
    f_{t} &=& h^{-1}(\tilde f_t), \\
    \tilde f_{t+1} &=& \omega + \sum_{i=1}^p A_{i}\tilde s_{t-i+1} + \sum_{j=1}^q B_{j}\tilde f_{t-j+1}
    \end{array}
    \right.
\end{equation*}
```

Notice that the change in parametrization changes the dynamics of the model. The GAS(1,1) for a Normal distribution with inverse scaling ``d = 1`` is equivalent to the GARCH(1, 1) model, but only on the original parameter, if you work with a different parametrization the model is no longer equivalent.


### Types of links

The abstract type `Link` subsumes any type of link that can be expressed.

```@docs
GAS.IdentityLink
GAS.LogLink
GAS.LogitLink
```

### Link functions

```@docs
GAS.link
GAS.unlink
GAS.jacobian_link
```

