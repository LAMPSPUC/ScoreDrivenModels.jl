abstract type Link end

struct ExponentialLink <: Link end

param_to_param_tilde(::Type{ExponentialLink}, param::T) where T = log(param)
param_tilde_to_param(::Type{ExponentialLink}, param_tilde::T) where T = exp(param_tilde)
jacobian_param_tilde(::Type{ExponentialLink}, param_tilde::T) where T = exp(param_tilde)

struct IdentityLink <: Link end

param_to_param_tilde(::Type{IdentityLink}, param::T) where T = param
param_tilde_to_param(::Type{IdentityLink}, param_tilde::T) where T = param_tilde
jacobian_param_tilde(::Type{IdentityLink}, param_tilde::T) where T = one(T)

# Poisson
param_to_param_tilde(poisson::Poisson, param::Vector{T}) where T = param_to_param_tilde.(ExponentialLink, param)
param_tilde_to_param(poisson::Poisson, param_tilde::Vector{T}) where T = param_tilde_to_param.(ExponentialLink, param_tilde)
jacobian_param_tilde(poisson::Poisson, param_tilde::Vector{T}) where T = Diagonal(jacobian_param_tilde.(ExponentialLink, param_tilde))

# Normal
function param_to_param_tilde(normal::Normal, param::Vector{T}) where T 
    return [
        param_to_param_tilde(IdentityLink, param[1])
        param_to_param_tilde(ExponentialLink, param[2])
    ]
end
function param_tilde_to_param(normal::Normal, param_tilde::Vector{T}) where T 
    return [
        param_tilde_to_param(IdentityLink, param_tilde[1])
        param_tilde_to_param(ExponentialLink, param_tilde[2])
    ]
end
function jacobian_param_tilde(normal::Normal, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_param_tilde(IdentityLink, param_tilde[1])
        jacobian_param_tilde(ExponentialLink, param_tilde[2])
    ])
end
