"""
    LogNormal

* Parametrization
parametrized in \\mu and \\sigma^2

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
function LogNormal end

function score!(score_til::Matrix{T}, y::T, ::Type{LogNormal}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = (log(y) - param[t, 1])/param[t, 2]
    score_til[t, 2] = -(0.5/param[t, 2]) * (1 - ((log(y) - param[t, 1])^2)/param[t, 2])
    return
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{LogNormal}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = 1/(param[t, 2])
    aux.fisher[2, 2] = 1/(2*(param[t, 2]^2))
    aux.fisher[2, 1] = 0
    aux.fisher[1, 2] = 0
    return
end

function log_likelihood(::Type{LogNormal}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = -0.5*log(2*pi)*n
    for t in 1:n
        loglik -= log(y[t] * sqrt(param[t, 2])) + 0.5*(log(y[t]) - param[t, 1])^2/param[t, 2]
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{LogNormal}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(IdentityLink, param[t, 1])
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{LogNormal}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(IdentityLink, param_tilde[t, 1])
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{LogNormal}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(IdentityLink, param[t, 1])
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    return
end

# utils 
function update_dist(::Type{LogNormal}, param::Matrix{T}, t::Int) where T
    # lognormal here is parametrized as sigma^2
    return LogNormal(param[t, 1], sqrt(param[t, 2]))
end 

function num_params(::Type{LogNormal})
    return 2
end