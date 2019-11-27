# Currently supported distributions
const DISTS = [
    Normal;
    Beta;
    Poisson;
    LogNormal;
    Gamma;
    Weibull
]

function score(y::T, D::Type{<:Distribution}, param::Vector{T}) where T
    return error("score not implemented for $D distribution")
end
function score(y::Int, D::Type{<:Distribution}, param::Vector{T}) where T
    return error("score not implemented for $D distribution")
end
function fisher_information(D::Type{<:Distribution}, param::Vector{T}) where T
    return error("fisher_information not implemented for $D distribution")
end
function param_to_param_tilde(D::Type{<:Distribution}, param::Vector{T}) where T 
    return error("param_to_param_tilde not implemented for $D distribution")
end
function param_tilde_to_param(D::Type{<:Distribution}, param_tilde::Vector{T}) where T 
    return error("param_tilde_to_param not implemented for $D distribution")
end
function jacobian_link(D::Type{<:Distribution}, param_tilde::Vector{T}) where T 
    return error("jacobian_link not implemented for $D distribution")
end
function update_dist(D::Type{<:Distribution}, param::Vector{T}) where T
    return error("update_dist not implemented for $D distribution")
end 
function num_params(D::Type{<:Distribution})
    return error("num_params not implemented for $D distribution")
end
