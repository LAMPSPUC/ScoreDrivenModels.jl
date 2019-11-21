"""
Proof somewhere 
parametrized in \\alpha and \\theta
"""
function score(y::T, ::Type{Weibull}, param::Vector{T}) where T
    return [
        (1/param[1]) + log(y/param[2]) - log(y/param[2]) * (y/param[2])^param[1] ;
        param[1] * (1/param[2]) * (y/param[2])^param[1] - (param[1]/param[2])
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Type{Weibull}, param::Vector{T}) where T
    return #TODO
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Weibull}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = zero(T)
    for i in 1:n
        loglik += log(param[i][1]) + (param[i][1] - 1) * log(y[i]) - param[i][1] * log(param[i][2]) - (y[i]/param[i][2])^param[i][1]
    end
    return -loglik
end

# Links
function param_to_param_tilde(::Type{Weibull}, param::Vector{T}) where T 
    return [
        param_to_param_tilde(ExponentialLink, param[1], zero(T));
        param_to_param_tilde(ExponentialLink, param[2], zero(T))
    ]
end
function param_tilde_to_param(::Type{Weibull}, param_tilde::Vector{T}) where T 
    return [
        param_tilde_to_param(ExponentialLink, param_tilde[1], zero(T));
        param_tilde_to_param(ExponentialLink, param_tilde[2], zero(T))
    ]
end
function jacobian_param_tilde(::Type{Weibull}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_param_tilde(ExponentialLink, param_tilde[1], zero(T));
        jacobian_param_tilde(ExponentialLink, param_tilde[2], zero(T))
    ])
end

# utils 
function update_dist(::Type{Weibull}, param::Vector{T}) where T
    # normal here is parametrized as sigma^2
    return Weibull(param[1], param[2])
end 

function num_params(::Type{Weibull})
    return 2
end