"""
    Gamma

* Parametrization
parametrized in \\alpha and \\theta

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
Gamma

function score!(score_til::Matrix{T}, y::T, ::Type{Gamma}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = log(y) - digamma(param[t, 1]) - log(param[t, 2])
    score_til[t, 2] = y/param[t, 2]^2 - param[t, 1]/param[t, 2]
    return 
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Gamma}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = trigamma(param[t, 1])
    aux.fisher[2, 2] = param[t, 1] / param[t, 2]^2
    aux.fisher[2, 1] = 1/param[t, 2]
    aux.fisher[1, 2] = 1/param[t, 2]
    return
end

function log_likelihood(::Type{Gamma}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = 0.0
    for t in 1:n
        loglik += (param[t, 1] - 1)*log(y[t]) - y[t]/param[t, 2] - loggamma(param[t, 1]) - param[t, 1]*log(param[t, 2])
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{Gamma}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{Gamma}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Gamma}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    return
end

# utils 
function update_dist(::Type{Gamma}, param::Matrix{T}, t::Int) where T
    small_threshold!(param, SMALL_NUM, t)
    return Gamma(param[t, 1], param[t, 2])
end 

function params_sdm(d::Gamma)
    return Distributions.params(d)
end

function num_params(::Type{Gamma})
    return 2
end
