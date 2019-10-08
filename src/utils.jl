function fill_psitilde!(gas_sarima::GAS_Sarima, psitilde::Vector{T}, unknowns_gas_sarima::Unknowns_GAS_Sarima) where T
    offset = 0
    # fill ω
    for i in unknowns_gas_sarima.ω
        offset += 1
        gas_sarima.ω[i] = psitilde[offset]
    end
    # fill A
    for (k, v) in unknowns_gas_sarima.A
        for i in v
            offset += 1
            gas_sarima.A[k][i] = psitilde[offset]
        end
    end
    # fill B
    for (k, v) in unknowns_gas_sarima.B
        for i in v
            offset += 1
            gas_sarima.B[k][i] = psitilde[offset]
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

function number_of_lags(gas_sarima::GAS_Sarima)
    return max(maximum(keys(gas_sarima.A)), maximum(keys(gas_sarima.B)))
end