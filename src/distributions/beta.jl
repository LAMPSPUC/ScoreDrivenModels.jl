"""
Proof somewhere 
parametrized in \\alpha and \\beta
"""
function score(y::T, ::Type{Beta}, param::Matrix{T}, t::Int) where T
    digamma_a_b = digamma(param[t, 1] + param[t, 2])
    return [
        log(y) + digamma_a_b - digamma(param[t, 1]);
        log(1 - y) + digamma_a_b - digamma(param[t, 2])
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Type{Beta}, param::Matrix{T}, t::Int) where T
    minus_trigamma_a_b = -trigamma(param[t, 1] + param[t, 2])

    return [
        trigamma(param[t, 1]) + minus_trigamma_a_b    minus_trigamma_a_b;
        minus_trigamma_a_b                         trigamma(param[t, 2]) + minus_trigamma_a_b
    ]
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Beta}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = 0.0
    for t in 1:n
        loglik += (param[t, 1] - 1)*log(y[t]) + (param[t, 2] - 1)*log(1 - y[t]) - logbeta(param[t, 1], param[t, 2])
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{Beta}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
end
function unlink!(param::Matrix{T}, ::Type{Beta}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
end
function jacobian_link(::Type{Beta}, param::Matrix{T}, t::Int) where T 
    return Diagonal([
        jacobian_link(LogLink, param[t, 1], zero(T));
        jacobian_link(LogLink, param[t, 2], zero(T))
    ])
end

# utils
function update_dist(::Type{Beta}, param::Matrix{T}, t::Int) where T
    small_threshold!(param, ZERO_ROUNDING, t)
    return Beta(param[t, 1], param[t, 2])
end 

function num_params(::Type{Beta})
    return 2
end