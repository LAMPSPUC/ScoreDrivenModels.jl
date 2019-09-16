# Poisson
"""
Proof somewhere
"""
function score(y, poisson::Poisson, param::Vector{T}) where T
    return [y/param[1] - 1]
end

"""
Proof somewhere
"""
function fisher_information(poisson::Poisson, param::Vector{T}) where T
    return (1/param[1])
end

"""
Proof somewhere
"""
function log_likelihood(poisson::Poisson, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in n
        loglik += y[i]*log(param[i][1]) - param[i][1] - log(factorial(y[i]))
    end
    return -loglik
end

# Links
param_to_param_tilde(poisson::Poisson, param::Vector{T}) where T = param_to_param_tilde.(ExponentialLink, param)
param_tilde_to_param(poisson::Poisson, param_tilde::Vector{T}) where T = param_tilde_to_param.(ExponentialLink, param_tilde)
jacobian_param_tilde(poisson::Poisson, param_tilde::Vector{T}) where T = Diagonal(jacobian_param_tilde.(ExponentialLink, param_tilde))