"""
Proof somewhere 
parametrized in \\mu and \\sigma^2
"""
function score!(score_til::Matrix{T}, y::T, ::Type{LogNormal}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = (log(y) - param[t, 1])/param[t, 2]
    score_til[t, 2] = -(0.5/param[t, 2]) * (1 - ((log(y) - param[t, 1])^2)/param[t, 2])
    return
end

"""
Proof somewhere
"""
function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{LogNormal}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = 1/(param[t, 2])
    aux.fisher[2, 2] = 1/(2*(param[t, 2]^2))
    aux.fisher[2, 1] = 0
    aux.fisher[1, 2] = 0
    return
end

"""
Proof somewhere
"""
function log_likelihood(::Type{LogNormal}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = -0.5*log(2*pi)*n
    for t in 1:n
        loglik -= log(y[t] * sqrt(param[t, 2])) + 0.5*(log(y[t]) - param[t, 1])^2/param[t, 2]
    end
    return -loglik
end

function default_links(::Type{LogNormal})
    return [IdentityLink(); LogLink(0.0)]
end

# utils 
function update_dist(::Type{LogNormal}, param::Matrix{T}, t::Int) where T
    # lognormal here is parametrized as sigma^2
    return LogNormal(param[t, 1], sqrt(param[t, 2]))
end 

function num_params(::Type{LogNormal})
    return 2
end