"""
    Beta
* Parametrization
* Score
* Fisher Information
* `time_varying_params` map.
* Default link
"""
Beta

function score!(score_til::Matrix{T}, y::T, ::Type{Beta}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = log(y) + digamma(param[t, 1] + param[t, 2]) - digamma(param[t, 1])
    score_til[t, 2] = log(1 - y) + digamma(param[t, 1] + param[t, 2]) - digamma(param[t, 2])
    return
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Beta}, param::Matrix{T}, t::Int) where T
    minus_trigamma_a_b = -trigamma(param[t, 1] + param[t, 2])
    aux.fisher[1, 1] = trigamma(param[t, 1]) + minus_trigamma_a_b
    aux.fisher[2, 2] = trigamma(param[t, 2]) + minus_trigamma_a_b
    aux.fisher[2, 1] = minus_trigamma_a_b
    aux.fisher[1, 2] = minus_trigamma_a_b
    return
end

function log_likelihood(::Type{Beta}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = 0.0
    for t in 1:n
        loglik += (param[t, 1] - 1)*log(y[t]) + (param[t, 2] - 1)*log(1 - y[t]) - logbeta(param[t, 1], param[t, 2])
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{Beta}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{Beta}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Beta}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    return
end

# utils
function update_dist(::Type{Beta}, param::Matrix{T}, t::Int) where T
    small_threshold!(param, SMALL_NUM, t)
    return Beta(param[t, 1], param[t, 2])
end 

function params_sdm(d::Beta)
    return Distributions.params(d)
end

function num_params(::Type{Beta})
    return 2
end