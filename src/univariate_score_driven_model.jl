export SDModel

export score_driven_recursion

mutable struct SDModel
    ω::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    dist::Distribution
    scaling

    function SDModel(ω::Vecor{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, dist::Distribution, scaling::Float64)
        return new(ω, A, B, dist, scaling)
    end
end

function score_driven_recursion(sd_model::SDModel, y::Vector{T}) where T
    return score_driven_recursion(sd_model, y, stationary_initial_params(sd_model))
end

function score_driven_recursion(sd_model::SDModel, y::Vector{T}, initial_param::Vector{T}) where T
    # Allocations
    n = length(y)
    param = Vector{Vector{T}}(undef, n + 1)
    param_tilde = Vector{Vector{T}}(undef, n + 1)

    # initial_values 
    param_tilde[1] = initial_param

    for i in 1:n
        param[i] = param_tilde_to_param(sd_model.dist, param_tilde[i])
        score_til = score_tilde(y[i], sd_model.dist, param[i], param_tilde[i], sd_model.scaling)
        update_param_tilde(param_tilde, sd_model.dist, sd_model.ω, sd_model.A, sd_model.B, score_til, i)
    end
    param[end] = param_tilde_to_param(sd_model.dist, param_tilde[end])

    return param
end

function update_param_tilde(param_tilde, dist, ω, A, B, score_til, i)
    param_tilde[i + 1] = ω + A*score_til + B*param_tilde[i]
    return 
end