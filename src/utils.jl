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

function find_unknowns(vec::Vector{T}) where T
    return findall(isnan, vec)
end

function find_unknowns(mat::Matrix{T}) where T
    return findall(isnan, vec(mat))
end

function check_model_estimated(len::Int)
    if len == 0
        println("Score Driven Model does not have unknowns.")
        return true
    end
    return false
end

function NaN2zero!(score_til::Vector{T}) where T
    for i in eachindex(score_til)
        if isnan(score_til[i])
            score_til[i] = zero(T) 
        end
    end
    return 
end

function big_threshold!(score_til::Vector{T}, threshold::T) where T
    for i in eachindex(score_til)
        if score_til[i] >= threshold
            score_til[i] = threshold 
        end
        if score_til[i] <= -threshold
            score_til[i] = -threshold 
        end
    end
    return 
end

function small_threshold!(score_til::Vector{T}, threshold::T) where T
    for i in eachindex(score_til)
        if score_til[i] <= threshold && score_til[i] >= 0
            score_til[i] = threshold 
        end
        if score_til[i] >= -threshold && score_til[i] <= 0
            score_til[i] = -threshold 
        end
    end
    return 
end

function update_dist(dist::Distribution, param::Vector{T}) where T
    error("not implemented")
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

function dim_unknowns(gas::GAS)
    return length(find_unknowns(gas))
end

function number_of_lags(gas::GAS)
    return max(maximum(keys(gas.A)), maximum(keys(gas.B)))
end