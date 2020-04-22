"""
    NegativeBinomial

* Parametrization
parametrized in r, p

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
NegativeBinomial

function score!(score_til::Matrix{T}, y::Int, ::Type{NegativeBinomial}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = digamma(y + param[t, 1]) - digamma(param[t, 1]) + log(param[t, 2])
    score_til[t, 2] = param[t, 1] / param[t, 2] - y / (one(T) - param[t, 2])
    return
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{NegativeBinomial}, param::Matrix{T}, t::Int) where T
    return error("Fisher information not implemented for NegativeBinomial distribution.")
end

function log_likelihood(::Type{NegativeBinomial}, y::Vector{Int}, param::Matrix{T}, n::Int) where T
    loglik = zero(T)
    for t in 1:n
        loglik += loggamma(y[t] + param[t, 1]) - logfactorial(y[t]) - loggamma(param[t, 1]) + 
                param[t, 1] * log(param[t, 2]) + y[t] * log(1 - param[t, 2])
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{NegativeBinomial}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    param_tilde[t, 2] = link(LogitLink, param[t, 2], zero(T), one(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{NegativeBinomial}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    param[t, 2] = unlink(LogitLink, param_tilde[t, 2], zero(T), one(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{NegativeBinomial}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    aux.jac[2] = jacobian_link(LogitLink, param[t, 2], zero(T), one(T))
    return
end

# utils
function update_dist(::Type{NegativeBinomial}, param::Matrix{T}, t::Int) where T
    return NegativeBinomial(param[t, 1], param[t, 2])
end 

function params_sdm(d::NegativeBinomial)
    return Distributions.params(d)
end

function num_params(::Type{NegativeBinomial})
    return 2
end