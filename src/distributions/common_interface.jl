# Currently supported distributions
const DISTS = [
    Normal;
    Beta;
    Poisson;
    LogNormal;
    Gamma;
    Weibull
]

export Normal, Beta, Poisson, LogNormal, Gamma, Weibull

function score!(score_til::Matrix{T}, y::T, ::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("score! not implemented for $D distribution")
end
function score!(score_til::Matrix{T}, y::Int, ::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("score! not implemented for $D distribution")
end
function fisher_information!(aux::AuxiliaryLinAlg{T},::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("fisher_information! not implemented for $D distribution")
end
function log_likelihood(::Type{<:Distribution}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    return error("log_likelihood not implemented for $D distribution")
end
function update_dist(::Type{<:Distribution}, param::Matrix{T}, t::Int) where T
    return error("update_dist not implemented for $D distribution")
end 
function num_params(D::Type{<:Distribution})
    return error("num_params not implemented for $D distribution")
end
