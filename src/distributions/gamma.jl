"""
Proof somewhere 
parametrized in \\alpha and \\theta
"""
function score!(score_til::Matrix{T}, y::T, ::Type{Gamma}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = log(y) - digamma(param[t, 1]) - log(param[t, 2])
    score_til[t, 2] = y/param[t, 2]^2 - param[t, 1]/param[t, 2]
    return 
end

"""
Proof somewhere
"""
function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Gamma}, param::Matrix{T}, t::Int) where T
    error("Fisher information not implemented for Gamma distribution.")
    aux.fisher[1, 1] = -trigamma(param[t, 1])
    aux.fisher[2, 2] = -2 * y / param[t, 2] ^ 3 + param[t, 1] / param[t, 2]^2
    aux.fisher[2, 1] = -1/param[t, 2]
    aux.fisher[1, 2] = -1/param[t, 2]
    return
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Gamma}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = 0.0
    for t in 1:n
        loglik += (param[t, 1] - 1)*log(y[t]) - y[t]/param[t, 2] - loggamma(param[t, 1]) - param[t, 1]*log(param[t, 2])
    end
    return -loglik
end

function default_links(::Type{Gamma})
    return [LogLink(0.0); LogLink(0.0)]
end

# utils 
function update_dist(::Type{Gamma}, param::Matrix{T}, t::Int) where T
    small_threshold!(param, SMALL_NUM, t)
    return Gamma(param[t, 1], param[t, 2])
end 

function num_params(::Type{Gamma})
    return 2
end
