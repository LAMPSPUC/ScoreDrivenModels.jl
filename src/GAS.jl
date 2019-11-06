export GAS

mutable struct GAS{D <: Distribution, T <: AbstractFloat} <: SDM
    ω::Vector{T}
    A::Dict{Int, Matrix{T}}
    B::Dict{Int, Matrix{T}}
    scaling::Real

    function GAS{D, T}(ω::Vector{Float64}, A::Dict{Int, Matrix{T}}, B::Dict{Int, Matrix{T}}, scaling::Real) where {D <: Distribution, T <: AbstractFloat}
        return new(ω, A, B, scaling)
    end
end

function create_ω(num_params::Int)
    return fill(NaN, num_params)
end

function create_lagged_matrix(lags::Vector{Int}, time_varying_params::Vector{Int}, zeros_params::Vector{T}) where T
    mat = Dict{Int, Matrix{T}}()
    for i in lags
        mat[i] = Diagonal(zeros_params)
        for k in time_varying_params
            mat[i][k, k] = NaN
        end
    end
    return mat
end

function GAS(p::Int, q::Int, D::Type{<:Distribution}, scaling::Real; 
             time_varying_params::Vector{Int} = collect(1:num_params(D)))

    # Vector of unkowns
    zeros_params = fill(0.0, num_params(D))
    ω = create_ω(num_params(D))
    
    # Create A and B
    A = create_lagged_matrix(collect(1:p), time_varying_params, zeros_params)
    B = create_lagged_matrix(collect(1:q), time_varying_params, zeros_params)

    return GAS{D, Float64}(ω, A, B, scaling)
end

function GAS(ps::Vector{Int}, qs::Vector{Int}, D::Type{<:Distribution}, scaling::Real; 
             time_varying_params::Vector{Int} = collect(1:num_params(D)))

    # Vector of unkowns
    zeros_params = fill(0.0, num_params(D))
    ω = create_ω(num_params(D))
    
    # Create A and B
    A = create_lagged_matrix(ps, time_varying_params, zeros_params)
    B = create_lagged_matrix(qs, time_varying_params, zeros_params)

    return GAS{D, Float64}(ω, A, B, scaling)
end

"""
The unknows parameters of a GAS model
Only for internal use
"""
mutable struct Unknowns_GAS <: Unknowns_SDM
    ω::Vector{Int}
    A::Dict{Int, Vector{Int}}
    B::Dict{Int, Vector{Int}}
end