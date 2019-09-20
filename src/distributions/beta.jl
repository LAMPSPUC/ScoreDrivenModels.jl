"""
Proof somewhere 
parametrized in \\alpha and \\beta
"""
function score(y::T, ::Beta, param::Vector{T}) where T
    digamma_a_b = digamma(param[1] + param[2])
    return [
        log(y) + digamma_a_b - digamma(param[1]);
        log(1 - y) + digamma_a_b - digamma(param[2])
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Beta, param::Vector{T}) where T
    minus_trigamma_a_b = -trigamma(param[1] + param[2])

    return [
        trigamma(param[1]) + minus_trigamma_a_b    minus_trigamma_a_b;
        minus_trigamma_a_b                         trigamma(param[2]) + minus_trigamma_a_b
    ]
end

"""
Proof somewhere
"""
function log_likelihood(::Beta, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in 1:n
        loglik += (param[i][1] - 1)*log(y[i]) + (param[i][2] - 1)*log(1 - y[i]) - log(beta(param[i][1], param[i][2]))
    end
    return -loglik
end

# Links
function param_to_param_tilde(::Beta, param::Vector{T}) where T 
    return [
        param_to_param_tilde(ExponentialLink, param[1]);
        param_to_param_tilde(ExponentialLink, param[2])
    ]
end
function param_tilde_to_param(::Beta, param_tilde::Vector{T}) where T 
    return [
        param_tilde_to_param(ExponentialLink, param_tilde[1]);
        param_tilde_to_param(ExponentialLink, param_tilde[2])
    ]
end
function jacobian_param_tilde(::Beta, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_param_tilde(ExponentialLink, param_tilde[1]);
        jacobian_param_tilde(ExponentialLink, param_tilde[2])
    ])
end