"""
Proof somewhere 
parametrized in \\alpha and \\theta
"""
function score!(score_til::Matrix{T}, y::T, ::Type{Weibull}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = (1/param[t, 1]) + log(y/param[t, 2]) * (1 - (y/param[t, 2])^param[t, 1]) 
    score_til[t, 2] = (param[t, 1]/param[t, 2]) * (((y/param[t, 2])^param[t, 1]) - 1)
end

"""
Proof somewhere
"""
function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Weibull}, param::Matrix{T}, t::Int) where T
    return error("Fisher information not implemented for Weibull distribution.")
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Weibull}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = zero(T)
    for t in 1:n
        loglik += log(param[t, 1]) + (param[t, 1] - 1) * log(y[t]) - param[t, 1] * log(param[t, 2]) - (y[t] / param[t, 2]) ^ param[t, 1]
    end
    return -loglik
end

function default_links(::Type{Weibull})
    return [LogLink(0.0); LogLink(0.0)]
end


# utils 
function update_dist(::Type{Weibull}, param::Matrix{T}, t::Int) where T
    # normal here is parametrized as sigma^2
    return Weibull(param[t, 1], param[t, 2])
end 

function num_params(::Type{Weibull})
    return 2
end