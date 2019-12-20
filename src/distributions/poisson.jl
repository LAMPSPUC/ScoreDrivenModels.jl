"""
Proof somewhere
"""
function score!(score_til::Matrix{T}, y::Int, ::Type{Poisson}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = y/param[t, 1] - 1
    return
end

"""
Proof somewhere
"""
function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{Poisson}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = 1/param[1]
end

"""
Proof somewhere
"""
function log_likelihood(::Type{Poisson}, y::Vector{Int}, param::Matrix{T}, n::Int) where T
    loglik = zero(T)
    for t in 1:n
        loglik += y[t] * log(param[t, 1]) - param[t, 1] - logfactorial(y[t])
    end
    return -loglik
end

function default_links(::Type{Poisson})
    return [IdentityLink()]
end

# utils
function update_dist(::Type{Poisson}, param::Matrix{T}, t::Int) where T
    return Poisson(param[t, 1])
end 

function num_params(::Type{Poisson})
    return 1
end