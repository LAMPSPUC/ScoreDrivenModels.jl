function check_model_estimated(len::Int)
    if len == 0
        println("Score Driven Model does not have unknowns.")
        return true
    end
    return false
end

# function NaN2zero!(v::Vector{T}) where T
#     for i in eachindex(v)
#         if isnan(v[i])
#             v[i] = zero(T) 
#         end
#     end
#     return 
# end

# function big_threshold!(v::Vector{T}, threshold::T) where T
#     for i in eachindex(v)
#         if v[i] >= threshold
#             v[i] = threshold 
#         end
#         if v[i] <= -threshold
#             v[i] = -threshold 
#         end
#     end
#     return 
# end

# function small_threshold!(v::Vector{T}, threshold::T) where T
#     for i in eachindex(v)
#         if v[i] <= threshold && v[i] >= 0
#             v[i] = threshold 
#         end
#         if v[i] >= -threshold && v[i] <= 0
#             v[i] = -threshold 
#         end
#     end
#     return 
# end

function NaN2zero!(m::Matrix{T}, i::Integer) where T
    for j in axes(m, 2)
        if isnan(m[i, j])
            m[i, j] = zero(T) 
        end
    end
    return 
end

function big_threshold!(m::Matrix{T}, threshold::T, i::Integer) where T
    for j in axes(m, 2)
        if m[i, j] >= threshold
            m[i, j] = threshold 
        end
        if m[i, j] <= -threshold
            m[i, j] = -threshold 
        end
    end
    return 
end

function small_threshold!(m::Matrix{T}, threshold::T, i::Integer) where T
    for j in axes(m, 2)
        if m[i, j] <= threshold && m[i, j] >= 0
            m[i, j] = threshold 
        end
        if m[i, j] >= -threshold && m[i, j] <= 0
            m[i, j] = -threshold 
        end
    end
    return 
end

function sample_observation(dist::Distribution)
    return rand(dist)
end

function find_unknowns(v::Vector{T}) where T
    return findall(isnan, v)
end

function find_unknowns(m::Matrix{T}) where T
    return findall(isnan, vec(m))
end