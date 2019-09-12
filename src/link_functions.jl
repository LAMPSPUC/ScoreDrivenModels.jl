abstract type Link end

struct ExponentialLink <: Link end

param_to_param_tilde(::Type{ExponentialLink}, param) = log.(param)
param_tilde_to_param(::Type{ExponentialLink}, param_tilde) = exp.(param_tilde)
jacobian_param_tilde(::Type{ExponentialLink}, param_tilde) = exp.(param_tilde)

# Poisson
param_to_param_tilde(poisson::Poisson, param) = param_to_param_tilde(ExponentialLink, param)
param_tilde_to_param(poisson::Poisson, param_tilde) = param_tilde_to_param(ExponentialLink, param_tilde)
jacobian_param_tilde(poisson::Poisson, param_tilde) = jacobian_param_tilde(ExponentialLink, param_tilde)