function fill_ω!(sd_model::SDModel, psi_tilde::Vector{T}, unknowns_ω::Vector{Int}, offset::Int) where T
    for (i, pos) in enumerate(unknowns_ω)
        sd_model.ω[pos] = psi_tilde[i + offset]
    end
    return 
end

function fill_A!(sd_model::SDModel, psi_tilde::Vector{T}, unknowns_A::Vector{Int}, offset::Int) where T
    for (i, pos) in enumerate(unknowns_A)
        sd_model.A[pos, pos] = psi_tilde[i + offset]
    end
    return 
end

function fill_B!(sd_model::SDModel, psi_tilde::Vector{T}, unknowns_B::Vector{Int}, offset) where T
    for (i, pos) in enumerate(unknowns_B)
        sd_model.B[pos, pos] = psi_tilde[i + offset]
    end
    return 
end

function fill_psitilde!(sd_model::SDModel, psitilde::Vector{T}, unknowns_ω::Vector{Int},
                        unknowns_A::Vector{Int}, unknowns_B::Vector{Int}) where T
    fill_ω!(sd_model, psitilde, unknowns_ω, 0)
    fill_A!(sd_model, psitilde, unknowns_A, length(unknowns_ω))
    fill_B!(sd_model, psitilde, unknowns_B, length(unknowns_ω) + length(unknowns_A))
    return 
end

function find_unknowns(array::Array{T}) where T
    return findall(isnan, array)
end

function find_unknowns(array::Matrix{T}) where T
    return findall(isnan, diag(array))
end

function check_model_estimated(len::Int)
    if len == 0
        println("Score Driven Model does not have unknowns.")
        return true
    end
    return false
end

function dimension_unkowns(sd_model::SDModel) where T
    return length(find_unknowns(sd_model.ω)) + length(find_unknowns(sd_model.A)) + 
           length(find_unknowns(sd_model.B))
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