# Currently supported distributions
const DISTS = [
    Normal;
    Beta;
    Poisson
]

function score(y::T, D::Type{<:Distribution}, param::Vector{T}) where T
    return error("score not implemented for $(typeof(D)) distribution")
end
function score(y::Int, D::Type{<:Distribution}, param::Vector{T}) where T
    return error("score not implemented for $(typeof(D)) distribution")
end
function fisher_information(D::Type{<:Distribution}, param::Vector{T}) where T
    return error("fisher information not implemented for $(typeof(D)) distribution")
end
