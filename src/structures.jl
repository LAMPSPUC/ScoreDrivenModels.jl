abstract type ScoreDrivenModel end

const SDM = ScoreDrivenModel

export GAS_Sarima

mutable struct GAS_Sarima <: SDM
    ω::Vector{Float64}
    A::Dict{Int, Matrix{Float64}}
    B::Dict{Int, Matrix{Float64}}
    dist::Distribution
    scaling::Real

    function GAS_Sarima(ω::Vector{Float64}, A::Dict{Int, Matrix{Float64}}, B::Dict{Int, Matrix{Float64}}, 
                        dist::Distribution, scaling::Real)
        return new(ω, A, B, dist, scaling)
    end
end

function create_ω(num_params::Int)
    return fill(NaN, num_params)
end

function create_lagged_matrix(lags::Vector{Int}, time_varying_params::Vector{Int}, zeros_params::Vector{T}) where T
    mat = Dict{Int, Matrix{Float64}}()
    for i in lags
        mat[i] = Diagonal(zeros_params)
        for k in time_varying_params
            mat[i][k, k] = NaN
        end
    end
    return mat
end

function GAS_Sarima(p::Int, q::Int, dist::Distribution, scaling::Real; 
                    time_varying_params::Vector{Int} = collect(1:num_params(dist)))

    # Vector of unkowns
    zeros_params = fill(0.0, num_params(dist))
    ω = create_ω(num_params(dist))
    
    # Create A and B
    A = create_lagged_matrix(collect(1:p), time_varying_params, zeros_params)
    B = create_lagged_matrix(collect(1:q), time_varying_params, zeros_params)

    return GAS_Sarima(ω, A, B, dist, scaling)
end

function GAS_Sarima(ps::Vector{Int}, qs::Vector{Int}, dist::Distribution, scaling::Real; 
                    time_varying_params::Vector{Int} = collect(1:num_params(dist)))

    # Vector of unkowns
    zeros_params = fill(0.0, num_params(dist))
    ω = create_ω(num_params(dist))
    
    # Create A and B
    A = create_lagged_matrix(ps, time_varying_params, zeros_params)
    B = create_lagged_matrix(qs, time_varying_params, zeros_params)

    return GAS_Sarima(ω, A, B, dist, scaling)
end

"""
The unknows parameters of a GAS_Sarima model
Only for internal use
"""
mutable struct Unknowns_GAS_Sarima
    ω::Vector{Int}
    A::Dict{Int, Vector{Int}}
    B::Dict{Int, Vector{Int}}
end

function find_unknowns(gas_sarima::GAS_Sarima)
    unknowns_A = Dict{Int, Vector{Int}}()
    unknowns_B = Dict{Int, Vector{Int}}()

    unknowns_ω = find_unknowns(gas_sarima.ω)

    for (k, v) in gas_sarima.A
        unknowns_A[k] = find_unknowns(v)
    end
    for (k, v) in gas_sarima.B
        unknowns_B[k] = find_unknowns(v)
    end
    return Unknowns_GAS_Sarima(unknowns_ω, unknowns_A, unknowns_B)
end

function length(unknowns::Unknowns_GAS_Sarima)
    len = length(values(unknowns.ω))
    for (k, v) in unknowns.A
        len += length(v)
    end
    for (k, v) in unknowns.B
        len += length(v)
    end
    return len
end

function dim_unknowns(gas_sarima::GAS_Sarima)
    return length(find_unknowns(gas_sarima))
end