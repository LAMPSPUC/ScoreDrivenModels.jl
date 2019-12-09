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

# Links
function link!(param_tilde::Matrix{T}, ::Type{Poisson}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(LogLink, param[t, 1], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{Poisson}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(LogLink, param_tilde[t, 1], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Poisson}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(LogLink, param[t, 1], zero(T))
    return
end

# utils
function update_dist(::Type{Poisson}, param::Matrix{T}, t::Int) where T
    return Poisson(param[t, 1])
end 

function num_params(::Type{Poisson})
    return 1
end