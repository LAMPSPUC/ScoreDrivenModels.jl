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

function sample_observation(dist::Distribution)
    return rand(dist)
end

function find_unknowns(vec::Vector{T}) where T
    return findall(isnan, vec)
end

function find_unknowns(mat::Matrix{T}) where T
    return findall(isnan, vec(mat))
end