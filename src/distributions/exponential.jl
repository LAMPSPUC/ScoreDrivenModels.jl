"""
    Exponential

* Parametrization
parametrized in \\lambda

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
Exponential

function score!(score_til::Matrix{T}, y::T, ::Type{Exponential}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = 1/param[t, 1] - y
    return
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Exponential}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = 1/(param[1]^2)
    return
end

function log_likelihood(::Type{Exponential}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = zero(T)
    for t in 1:n
        loglik += log(param[t, 1]) - param[t, 1]*y[t]
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{Exponential}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{Exponential}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Exponential}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    return
end

# utils
# Exponential in Distributions.jl is modeled in terms of ``\theta`` which is ``1/\\lambda``
function update_dist(::Type{Exponential}, param::Matrix{T}, t::Int) where T
    return Exponential(1/param[t, 1])
end 

function num_params(::Type{Exponential})
    return 1
end