"""
    BetaFourParameters
* Parametrization a, c, \\alpha, \\beta
* Score
* Fisher Information
* `time_varying_params` map.
* Default link

Right now when estimating a BetaFourParameters model is recomended to provide fixed parameters a and c, dont
with the code:

```julia
gas_beta_4 = ScoreDrivenModel([1, 2, 11, 12], [1, 2, 11, 12], BetaFourParameters, 0.0; time_varying_params=[3])
gas_beta_4.ω[1] = minimum(y) - 0.1*std(y) # parameter a
gas_beta_4.ω[2] = maximum(y) + 0.1*std(y) # parameter c
```

This code is a simple and not very accurate heuristic on how to estimate the a and c parameters. If you have estimated
using maximum likelihood it will be probably better

The fact occurs because if in the optimization the gradient makes `c < maximum(y)` or `a > minimum(y)` then it
is impossible that y comes from the BetaFourParameters distribution leading to DomainErrors
"""
BetaFourParameters

function score!(score_til::Matrix{T}, y::T, ::Type{BetaFourParameters}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = -(param[t, 3] - 1)/(y - param[t, 1]) + (param[t, 4] + param[t, 3] - 1)/(param[t, 2] - param[t, 1])
    score_til[t, 2] = (param[t, 4] - 1)/(param[t, 2] - y) - (param[t, 4] + param[t, 3] - 1)/(param[t, 2] - param[t, 1])
    score_til[t, 3] = log(y - param[t, 1]) - log(param[t, 2] - param[t, 1]) +
                        digamma(param[t, 3] + param[t, 4]) - digamma(param[t, 3])
    score_til[t, 4] = log(param[t, 2] - y) - log(param[t, 2] - param[t, 1]) +
                        digamma(param[t, 3] + param[t, 4]) - digamma(param[t, 4])
    return
end

function log_likelihood(::Type{BetaFourParameters}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = 0.0
    for t in 1:n
        loglik += (param[t, 3] - 1)*log(y[t] - param[t, 1]) + (param[t, 4] - 1) * log(param[t, 2] - y[t]) -
                  (param[t, 3] + param[t, 4] - 1) * log(param[t, 2] - param[t, 1]) - logbeta(param[t, 3], param[t, 4])
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{BetaFourParameters}, param::Matrix{T}, t::Int) where T
    param_tilde[t, 1] = link(IdentityLink, param[t, 1])
    param_tilde[t, 2] = link(IdentityLink, param[t, 2])
    param_tilde[t, 3] = link(LogLink, param[t, 3], zero(T))
    param_tilde[t, 4] = link(LogLink, param[t, 4], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{BetaFourParameters}, param_tilde::Matrix{T}, t::Int) where T
    param[t, 1] = unlink(IdentityLink, param_tilde[t, 1])
    param[t, 2] = unlink(IdentityLink, param_tilde[t, 2])
    param[t, 3] = unlink(LogLink, param_tilde[t, 3], zero(T))
    param[t, 4] = unlink(LogLink, param_tilde[t, 4], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{BetaFourParameters}, param::Matrix{T}, t::Int) where T
    aux.jac[1] = jacobian_link(IdentityLink, param[t, 1])
    aux.jac[2] = jacobian_link(IdentityLink, param[t, 2])
    aux.jac[3] = jacobian_link(LogLink, param[t, 3], zero(T))
    aux.jac[4] = jacobian_link(LogLink, param[t, 4], zero(T))
    return
end

# utils
function update_dist(::Type{BetaFourParameters}, param::Matrix{T}, t::Int) where T
    small_threshold!(param[:, 3:4], SMALL_NUM, t)
    beta = Beta(param[t, 3], param[t, 4])
    return AffineDistribution(param[t, 1], param[t, 2] - param[t, 1], beta)
end

function params_sdm(d::BetaFourParameters)
    return (d.μ, d.σ + d.μ, Distributions.params(d.ρ)...)
end

function num_params(::Type{BetaFourParameters})
    return 4
end
