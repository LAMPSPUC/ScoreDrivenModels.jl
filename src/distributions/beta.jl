"""
Proof somewhere 
parametrized in \\alpha and \\beta
"""
function score!(score_til::Matrix{T}, y::T, ::Type{Beta}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = log(y) + digamma(param[t, 1] + param[t, 2]) - digamma(param[t, 1])
    score_til[t, 2] = log(1 - y) + digamma(param[t, 1] + param[t, 2]) - digamma(param[t, 2])
    return
end

"""
Proof somewhere
"""
function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Beta}, param::Matrix{T}, t::Int) where T
    minus_trigamma_a_b = -trigamma(param[t, 1] + param[t, 2])
    aux.fisher[1, 1] = trigamma(param[t, 1]) + minus_trigamma_a_b
    aux.fisher[2, 2] = trigamma(param[t, 2]) + minus_trigamma_a_b
    aux.fisher[2, 1] = minus_trigamma_a_b
    aux.fisher[1, 2] = minus_trigamma_a_b
    return
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

function default_links(::Type{Beta})
    return [LogLink(0.0); LogLink(0.0)]
end

# utils
function update_dist(::Type{Beta}, param::Matrix{T}, t::Int) where T
    small_threshold!(param, SMALL_NUM, t)
    return Beta(param[t, 1], param[t, 2])
end 

function num_params(::Type{Beta})
    return 2
end