abstract type Link end

struct IdentityLink <: Link end

param_to_param_tilde(::Type{IdentityLink}, param::T) where T = param
param_tilde_to_param(::Type{IdentityLink}, param_tilde::T) where T = param_tilde
jacobian_param_tilde(::Type{IdentityLink}, param_tilde::T) where T = one(T)

struct ExponentialLink <: Link end

param_to_param_tilde(::Type{ExponentialLink}, param::T) where T = log(param)
param_tilde_to_param(::Type{ExponentialLink}, param_tilde::T) where T = exp(param_tilde)
jacobian_param_tilde(::Type{ExponentialLink}, param_tilde::T) where T = exp(param_tilde)

struct LogitLink <: Link end

