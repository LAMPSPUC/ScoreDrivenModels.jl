function fill_ω!(sd_model::SDModel, ω_tilde::Vector{T}) where T
    for i in eachindex(ω_tilde)
        sd_model.ω[i] = ω_tilde[i]
    end
    return 
end

function fill_A!(sd_model::SDModel, A_tilde::Vector{T}) where T
    for i in eachindex(A_tilde)
        sd_model.A[i, i] = A_tilde[i]
    end
    return 
end

function fill_B!(sd_model::SDModel, B_tilde::Vector{T}) where T
    for i in eachindex(B_tilde)
        sd_model.B[i, i] = B_tilde[i]
    end
    return 
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