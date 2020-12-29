"""
    BetaLocationScale

* Parametrization \\mu, \\sigma, \\alpha, \\beta

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
BetaLocationScale


# Remember for all the code that param[t, 2] is c - a
# and param[t, 1] is a so to get c one must make 
# param[t, 2] + param[t, 1]
function score!(score_til::Matrix{T}, y::T, ::Type{BetaLocationScale}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = -(param[t, 3] - 1)/(y - param[t, 1]) + (param[t, 4] + param[t, 3] - 1)/(param[t, 2] - param[t, 1])
    score_til[t, 2] = (param[t, 4] - 1)/(param[t, 2] - y) - (param[t, 4] + param[t, 3] - 1)/(param[t, 2] - param[t, 1])
    score_til[t, 3] = log(y - param[t, 1]) - log(param[t, 2] - param[t, 1]) + 
                        digamma(param[t, 3] + param[t, 4]) - digamma(param[t, 3])
    score_til[t, 4] = log(param[t, 2] - y) - log(param[t, 2] - param[t, 1]) + 
                        digamma(param[t, 3] + param[t, 4]) - digamma(param[t, 4])
    return
end

function log_likelihood(::Type{BetaLocationScale}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = 0.0
    for t in 1:n
        loglik += (param[t, 3] - 1)*log(y[t] - param[t, 1]) + (param[t, 4] - 1) * log(param[t, 2] - y[t]) - 
                  (param[t, 3] + param[t, 4] - 1) * log(param[t, 2] - param[t, 1]) - logbeta(param[t, 3], param[t, 4])
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{BetaLocationScale}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{BetaLocationScale}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{BetaLocationScale}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    return
end

# utils
function update_dist(::Type{BetaLocationScale}, param::Matrix{T}, t::Int) where T
    small_threshold!(param[:, 3:4], SMALL_NUM, t)
    beta = Beta(param[t, 3], param[t, 4])
    return LocationScale(param[t, 1], param[t, 2] - param[t, 1], beta)
end 

function params_sdm(d::BetaLocationScale)
    return (d.μ, d.σ + d.μ, Distributions.params(d.ρ)...)
end

function num_params(::Type{BetaLocationScale})
    return 4
end