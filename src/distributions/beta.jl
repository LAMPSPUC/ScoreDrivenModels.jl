"""
Proof somewhere 
parametrized in \\alpha and \\beta
"""
function score(y::T, ::Type{Beta}, param::Vector{T}) where T
    digamma_a_b = digamma(param[1] + param[2])
    return [
        log(y) + digamma_a_b - digamma(param[1]);
        log(1 - y) + digamma_a_b - digamma(param[2])
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Type{Beta}, param::Vector{T}) where T
    minus_trigamma_a_b = -trigamma(param[1] + param[2])

    return [
        trigamma(param[1]) + minus_trigamma_a_b    minus_trigamma_a_b;
        minus_trigamma_a_b                         trigamma(param[2]) + minus_trigamma_a_b
    ]
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Beta}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in 1:n
        loglik += (param[i][1] - 1)*log(y[i]) + (param[i][2] - 1)*log(1 - y[i]) - logbeta(param[i][1], param[i][2])
    end
    return -loglik
end

# Links
function link(::Type{Beta}, param::Vector{T}) where T 
    return [
        link(LogLink, param[1], zero(T));
        link(LogLink, param[2], zero(T))
    ]
end
function unlink(::Type{Beta}, param_tilde::Vector{T}) where T 
    return [
        unlink(LogLink, param_tilde[1], zero(T));
        unlink(LogLink, param_tilde[2], zero(T))
    ]
end
function jacobian_link(::Type{Beta}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_link(LogLink, param_tilde[1], zero(T));
        jacobian_link(LogLink, param_tilde[2], zero(T))
    ])
end

# utils
function update_dist(::Type{Beta}, param::Vector{T}) where T
    small_threshold!(param, T(1e-8))
    return Beta(param[1], param[2])
end 

function num_params(::Type{Beta})
    return 2
end