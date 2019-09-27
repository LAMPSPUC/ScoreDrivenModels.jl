"""
Proof somewhere
"""
function score(y::Int, ::Type{Poisson}, param::Vector{T}) where T
    return [y/param[1] - 1]
end

"""
Proof somewhere
"""
function fisher_information(::Type{Poisson}, param::Vector{T}) where T
    return (1/param[1])
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Poisson}, y::Vector{Int}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in 1:n
        loglik += y[i]*log(param[i][1]) - param[i][1] - lfactorial(y[i])
    end
    return -loglik
end

# Links
param_to_param_tilde(::Poisson, param::Vector{T}) where T = param_to_param_tilde.(ExponentialLink, param)
param_tilde_to_param(::Poisson, param_tilde::Vector{T}) where T = param_tilde_to_param.(ExponentialLink, param_tilde)
jacobian_param_tilde(::Poisson, param_tilde::Vector{T}) where T = Diagonal(jacobian_param_tilde.(ExponentialLink, param_tilde))

# utils
function update_dist(D::Type{Poisson}, param::Vector{T}) where T
    return Poisson(param[1])
end 

function num_params(::Type{Poisson})
    return 1
end