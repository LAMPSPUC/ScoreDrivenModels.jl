export ScoreDrivenModel

mutable struct ScoreDrivenModel{D <: Distribution, T <: AbstractFloat}
    ω::Vector{T}
    A::Dict{Int, Matrix{T}}
    B::Dict{Int, Matrix{T}}
    scaling::Real
end

function deepcopy(gas::ScoreDrivenModel{D, T}) where {D, T}
    return ScoreDrivenModel{D, T}(deepcopy(gas.ω), deepcopy(gas.A), deepcopy(gas.B), deepcopy(gas.scaling))
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

function ScoreDrivenModel(p::Int, q::Int, D::Type{<:Distribution}, scaling::Real;
             time_varying_params::Vector{Int} = collect(1:num_params(D)))

    # Vector of unkowns
    zeros_params = fill(0.0, num_params(D))
    ω = create_ω(num_params(D))

    # Create A and B
    A = create_lagged_matrix(collect(1:p), time_varying_params, zeros_params)
    B = create_lagged_matrix(collect(1:q), time_varying_params, zeros_params)

    return ScoreDrivenModel{D, Float64}(ω, A, B, scaling)
end

function ScoreDrivenModel(ps::Vector{Int}, qs::Vector{Int}, D::Type{<:Distribution}, scaling::Real;
             time_varying_params::Vector{Int} = collect(1:num_params(D)))

    # Vector of unkowns
    zeros_params = fill(0.0, num_params(D))
    ω = create_ω(num_params(D))

    # Create A and B
    A = create_lagged_matrix(ps, time_varying_params, zeros_params)
    B = create_lagged_matrix(qs, time_varying_params, zeros_params)

    return ScoreDrivenModel{D, Float64}(ω, A, B, scaling)
end

"""
    ScoreDrivenModel

The constructor of a score-driven model. The model receives the lag structure, the
distribution and the scaling. You can define the lag structure in two different
ways, either by passing integers `p` and `q` to add all lags from `1` to `p` and `1` to `q` or by
passing vectors of integers `ps` and `qs` containing the desired lags. Once you build
the model all of the unknown parameters that must be estimated are represented as `NaN`.

```jldoctest
# Passing p and q
julia> ScoreDrivenModel(2, 2, LogNormal, 0.5)
ScoreDrivenModel{LogNormal,Float64}([NaN, NaN], Dict(2=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), Dict(2=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), 0.5)

# Passing ps and qs
julia> ScoreDrivenModel([1, 12], [1, 12], Gamma, 0.0)
ScoreDrivenModel{Gamma,Float64}([NaN, NaN], Dict(12=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), Dict(12=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), 0.0)
```

If you don't want all the parameters to be considered time-varying you can express it
through the keyword argument `time_varying_params`, there you should pass a vector
containing a number that represents which parameter should be time-varying. As an example
in the Normal distribution `time_varying_params = [1]` indicates that only ``\\mu`` is
time-varying. You can find the table with the dictionary (number => parameter) in the
section [ScoreDrivenModels distributions](@ref).

```jldoctest
julia> ScoreDrivenModel([1, 12], [1, 12], Normal, 1.0; time_varying_params = [1])
ScoreDrivenModel{Normal,Float64}([NaN, NaN], Dict(12=>[NaN 0.0; 0.0 0.0],1=>[NaN 0.0; 0.0 0.0]), Dict(12=>[NaN 0.0; 0.0 0.0],1=>[NaN 0.0; 0.0 0.0]), 1.0)
```
"""
function ScoreDrivenModel end

function number_of_lags(gas::ScoreDrivenModel)
    return max(maximum(keys(gas.A)), maximum(keys(gas.B)))
end

"""
    Unknowns

Structure that stores the positions of the parameters to be estimated in the [`ScoreDrivenModel`](@ref).
"""
mutable struct Unknowns
    ω::Vector{Int}
    A::Dict{Int, Vector{Int}}
    B::Dict{Int, Vector{Int}}
end

function fill_psitilde!(gas::ScoreDrivenModel, psitilde::Vector{T}, unknowns::Unknowns) where T
    offset = 0
    # fill ω
    for i in unknowns.ω
        offset += 1
        @inbounds gas.ω[i] = psitilde[offset]
    end
    # fill A
    for (k, v) in unknowns.A
        for i in v
            offset += 1
            @inbounds gas.A[k][i] = psitilde[offset]
        end
    end
    # fill B
    for (k, v) in unknowns.B
        for i in v
            offset += 1
            @inbounds gas.B[k][i] = psitilde[offset]
        end
    end
    return
end

function find_unknowns(gas::ScoreDrivenModel)
    unknowns_A = Dict{Int, Vector{Int}}()
    unknowns_B = Dict{Int, Vector{Int}}()

    unknowns_ω = find_unknowns(gas.ω)

    for (k, v) in gas.A
        unknowns_A[k] = find_unknowns(v)
    end
    for (k, v) in gas.B
        unknowns_B[k] = find_unknowns(v)
    end
    return Unknowns(unknowns_ω, unknowns_A, unknowns_B)
end

function dim_unknowns(gas::ScoreDrivenModel)
    return length(find_unknowns(gas))
end

function length(unknowns::Unknowns)
    len = length(values(unknowns.ω))
    for (k, v) in unknowns.A
        len += length(v)
    end
    for (k, v) in unknowns.B
        len += length(v)
    end
    return len
end

function log_lik(psitilde::Vector{T}, y::Vector{T}, gas::ScoreDrivenModel{D, T},
                 initial_params::Matrix{T}, unknowns::Unknowns, n::Int) where {D, T}

    # Use the unkowns vectors to fill the right positions
    fill_psitilde!(gas, psitilde, unknowns)

    if isnan(initial_params[1]) # Means default stationary initialization
        params = score_driven_recursion(gas, y)
    else
        params = score_driven_recursion(gas, y; initial_params = initial_params)
    end

    return log_likelihood(D, y, params, n) / length(y)
end
