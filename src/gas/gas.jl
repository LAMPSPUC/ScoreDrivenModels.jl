export GAS

mutable struct GAS{D <: Distribution, T <: AbstractFloat} <: SDM{D, T}
    ω::Vector{T}
    A::Dict{Int, Matrix{T}}
    B::Dict{Int, Matrix{T}}
    scaling::Real
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

function number_of_lags(gas::GAS)
    return max(maximum(keys(gas.A)), maximum(keys(gas.B)))
end

"""
The unknows parameters of a GAS model
Only for internal use
    
Every unknonws must define
fill_psitilde
find_unknowns
dim_unknowns
length
"""
mutable struct Unknowns_GAS <: Unknowns_SDM
    ω::Vector{Int}
    A::Dict{Int, Vector{Int}}
    B::Dict{Int, Vector{Int}}
end

function fill_psitilde!(gas::GAS, psitilde::Vector{T}, unknowns::Unknowns_GAS) where T
    offset = 0
    # fill ω
    for i in unknowns.ω
        offset += 1
        gas.ω[i] = psitilde[offset]
    end
    # fill A
    for (k, v) in unknowns.A
        for i in v
            offset += 1
            gas.A[k][i] = psitilde[offset]
        end
    end
    # fill B
    for (k, v) in unknowns.B
        for i in v
            offset += 1
            gas.B[k][i] = psitilde[offset]
        end
    end
    return 
end

function find_unknowns(gas::GAS)
    unknowns_A = Dict{Int, Vector{Int}}()
    unknowns_B = Dict{Int, Vector{Int}}()

    unknowns_ω = find_unknowns(gas.ω)

    for (k, v) in gas.A
        unknowns_A[k] = find_unknowns(v)
    end
    for (k, v) in gas.B
        unknowns_B[k] = find_unknowns(v)
    end
    return Unknowns_GAS(unknowns_ω, unknowns_A, unknowns_B)
end

function dim_unknowns(gas::GAS)
    return length(find_unknowns(gas))
end

function length(unknowns::Unknowns_GAS)
    len = length(values(unknowns.ω))
    for (k, v) in unknowns.A
        len += length(v)
    end
    for (k, v) in unknowns.B
        len += length(v)
    end
    return len
end

function log_lik(psitilde::Vector{T}, y::Vector{T}, gas::GAS{D, T}, 
                 initial_params::Vector{Vector{T}}, unknowns::Unknowns_GAS, n::Int) where {D, T}
    
    # Use the unkowns vectors to fill the right positions
    fill_psitilde!(gas, psitilde, unknowns)

    if isnan(initial_params[1][1]) # Means default stationary initialization
        params = score_driven_recursion(gas, y)
    else
        params = score_driven_recursion(gas, y, initial_params)
    end

    return log_likelihood(D, y, params, n)
end