"""
Proof somewhere 
parametrized in \\alpha and \\theta
"""
function score!(score_til::Matrix{T}, y::T, ::Type{Gamma}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = log(y) - digamma(param[t, 1]) - log(param[t, 2])
    score_til[t, 1] = y/param[t, 2]^2 - param[t, 1]/param[t, 2]
    return 
end

"""
Proof somewhere
"""
function fisher_information(::Type{Gamma}, param::Vector{T}) where T
    [
        -trigamma(param[1])     -1/param[2];
        -1/param[2]             -2*y/param[2]^3 + param[1]/param[2]^2
    ]
end

"""
Proof somewhere
"""
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
function jacobian_link!(aux::AuxiliaryStruct{T}, ::Type{Gamma}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    return
end

# utils 
function update_dist(::Type{Gamma}, param::Matrix{T}, t::Int) where T
    return Gamma(param[t, 1], param[t, 2])
end 

function num_params(::Type{Gamma})
    return 2
end
