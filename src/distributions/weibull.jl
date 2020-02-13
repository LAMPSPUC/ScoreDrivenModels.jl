"""
    LogNormal

* Paramaterization
parametrized in \\alpha and \\theta

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
function Weibull end

function score!(score_til::Matrix{T}, y::T, ::Type{Weibull}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = (1/param[t, 1]) + log(y/param[t, 2]) * (1 - (y/param[t, 2])^param[t, 1]) 
    score_til[t, 2] = (param[t, 1]/param[t, 2]) * (((y/param[t, 2])^param[t, 1]) - 1)
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Weibull}, param::Matrix{T}, t::Int) where T
    return error("Fisher information not implemented for Weibull distribution.")
end

function log_likelihood(::Type{Weibull}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = zero(T)
    for t in 1:n
        loglik += log(param[t, 1]) + (param[t, 1] - 1) * log(y[t]) - param[t, 1] * log(param[t, 2]) - (y[t] / param[t, 2]) ^ param[t, 1]
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{Weibull}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{Weibull}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Weibull}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    return
end

# utils 
function update_dist(::Type{Weibull}, param::Matrix{T}, t::Int) where T
    # normal here is parametrized as sigma^2
    return Weibull(param[t, 1], param[t, 2])
end 

function num_params(::Type{Weibull})
    return 2
end