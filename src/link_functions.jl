abstract type Link end

struct IdentityLink <: Link end

param_to_param_tilde(::Type{IdentityLink}, param::T) where T = param
param_tilde_to_param(::Type{IdentityLink}, param_tilde::T) where T = param_tilde
jacobian_param_tilde(::Type{IdentityLink}, param_tilde::T) where T = one(T)

struct ExponentialLink <: Link end

param_to_param_tilde(::Type{ExponentialLink}, param::T, lower_bound::T) where T = log(param - lower_bound)
param_tilde_to_param(::Type{ExponentialLink}, param_tilde::T, lower_bound::T) where T = exp(param_tilde) + lower_bound
jacobian_param_tilde(::Type{ExponentialLink}, param_tilde::T, lower_bound::T) where T = exp(param_tilde)

struct LogitLink <: Link end

const LINKS = [
    IdentityLink;
    ExponentialLink
]