"""
    Chisq

* Parametrization
parametrized in k

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
function Chisq end

function score!(score_til::Matrix{T}, y::T, ::Type{Chisq}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = (log(y) - log(2) - digamma(param[t, 1]/2))/2
    return 
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Chisq}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = trigamma(param[t, 1]/2)/4
    return
end

function log_likelihood(::Type{Chisq}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = 0.0
    for t in 1:n
        loglik += (param[t, 1]/2 - 1) * log(y[t]) - y[t]/2 - (param[t, 1]*log(2))/2 - loggamma(param[t, 1]/2)
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{Chisq}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{Chisq}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Chisq}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    return
end

# utils 
function update_dist(::Type{Chisq}, param::Matrix{T}, t::Int) where T
    small_threshold!(param, SMALL_NUM, t)
    return Chisq(param[t, 1])
end 

function num_params(::Type{Chisq})
    return 1
end
