# Currently supported distributions
const DISTS = [
    Beta,
    Chi,
    Chisq,
    Exponential,
    Gamma,
    LocationScaleTDist,
    LogitNormal,
    LogNormal,
    Normal,
    Poisson,
    TDist,
    Weibull
]

export Beta,
    Chi,
    Chisq,
    Exponential,
    Gamma,
    LocationScaleTDist,
    LogitNormal,
    LogNormal,
    Normal,
    Poisson,
    TDist,
    Weibull

# Query params for generic distribution
params_sdm(d::Distribution) = Distributions.params(d)
# Query params for all LocationScale distributions
params_sdm(d::LocationScale) = (d.μ, d.σ, Distributions.params(d.ρ)...)

function score!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("score! not implemented for $D distribution")
end
function score!(score_til::Matrix{T}, y::Int, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("score! not implemented for $D distribution")
end
function fisher_information!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("fisher_information! not implemented for $D distribution")
end
function log_likelihood(D::Type{<:Distribution}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    return error("log_likelihood not implemented for $D distribution")
end
function link!(param_tilde::Matrix{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T 
    return error("link! not implemented for $D distribution")
end
function unlink!(param::Matrix{T}, D::Type{<:Distribution}, param_tilde::Matrix{T}, t::Int) where T 
    return error("unlink! not implemented for $D distribution")
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T 
    return error("jacobian_link! not implemented for $D distribution")
end
function update_dist(D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("update_dist not implemented for $D distribution")
end 
function num_params(D::Type{<:Distribution})
    return error("num_params not implemented for $D distribution")
end
