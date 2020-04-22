# Currently supported distributions
const DISTS = [
    Beta,
    BetaLocationScale,
    Chi,
    Chisq,
    Exponential,
    Gamma,
    LogitNormal,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    TDist,
    TDistLocationScale,
    Weibull
]

export Beta,
    BetaLocationScale,
    Chi,
    Chisq,
    Exponential,
    Gamma,
    LogitNormal,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    TDist,
    TDistLocationScale,
    Weibull

"""
    score!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T

Fill `score_til` with the score of distribution `D` with parameters `param[:, t]` considering
the observation `y`.
"""
function score!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("score! not implemented for $D distribution")
end

function score!(score_til::Matrix{T}, y::Int, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("score! not implemented for $D distribution")
end

"""
    fisher_information!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T

Fill `aux` with the fisher information of distribution `D` with parameters `param[:, t]`.
"""
function fisher_information!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("fisher_information! not implemented for $D distribution")
end

"""
    log_likelihood(D::Type{<:Distribution}, y::Vector{T}, param::Matrix{T}, n::Int) where T

Evaluate the log-likelihood of the distribution `D` considering the time-varying parameters
`param` and the observations `y`.
"""
function log_likelihood(D::Type{<:Distribution}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    return error("log_likelihood not implemented for $D distribution")
end

"""
    link!(param_tilde::Matrix{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T 

Fill `param_tilde` after the unlinking procedure of `param`.
"""
function link!(param_tilde::Matrix{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T 
    return error("link! not implemented for $D distribution")
end

"""
    unlink!(param::Matrix{T}, D::Type{<:Distribution}, param_tilde::Matrix{T}, t::Int) where T 

Fill `param` after the unlinking procedure of `param_tilde`.
"""
function unlink!(param::Matrix{T}, D::Type{<:Distribution}, param_tilde::Matrix{T}, t::Int) where T 
    return error("unlink! not implemented for $D distribution")
end

"""
    jacobian_link!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T 

Write the jacobian of the link map in `aux`.
"""
function jacobian_link!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T 
    return error("jacobian_link! not implemented for $D distribution")
end

"""
    update_dist(D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T

Create a new distribution from Distributions.jl based on the parametrization used in
ScoreDrivenModels.jl.
"""
function update_dist(D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("update_dist not implemented for $D distribution")
end

"""
    params_sdm(d::Distribution)

Recover the parametrization used in ScoreDrivenModels.jl based on a Distribution from
Distributions.jl.
"""
function params_sdm(d::Distribution)
    return error("params_sdm not implemented for $d distribution")
end 

"""
    num_params(D::Type{<:Distribution})

Number of parameters of a given distribution.
"""
function num_params(D::Type{<:Distribution})
    return error("num_params not implemented for $D distribution")
end
