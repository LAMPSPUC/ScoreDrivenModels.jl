export SDModel

export score_driven_recursion

mutable struct SDModel
    ω::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    dist::Distribution
    scaling

    function SDModel(ω::Vector{Float64}, A::Matrix{Float64}, B::Matrix{Float64}, dist::Distribution, scaling::Float64)
        return new(ω, A, B, dist, scaling)
    end
end

function score_driven_recursion(sd_model::SDModel, observations::Vector{T}) where T
    return score_driven_recursion(sd_model, observations, stationary_initial_params(sd_model))
end

function score_driven_recursion(sd_model::SDModel, observations::Vector{T}, initial_param::Vector{T}) where T
    # Allocations
    n = length(observations)
    param = Vector{Vector{T}}(undef, n + 1)
    param_tilde = Vector{Vector{T}}(undef, n + 1)

    # initial_values 
    param_tilde[1] = initial_param

    for i in 1:n
        univariate_score_driven_update!(param, param_tilde, observations[i], sd_model, i)
    end
    update_param!(param, param_tilde, sd_model.dist, n + 1 #=end=#)

    return param
end

function univariate_score_driven_update!(param::Vector{Vector{T}}, param_tilde::Vector{Vector{T}}, 
                                         observation::T, sd_model::SDModel, i::Int) where T
    # update param 
    update_param!(param, param_tilde, sd_model.dist, i)
    # evaluate score
    score_til = score_tilde(observation, sd_model.dist, param[i], param_tilde[i], sd_model.scaling)
    # update param_tilde
    update_param_tilde!(param_tilde, sd_model.dist, sd_model.ω, sd_model.A, sd_model.B, score_til, i)
    return 
end

function update_param!(param::Vector{Vector{T}}, param_tilde::Vector{Vector{T}}, dist::Distribution, i::Int) where T
    param[i] = param_tilde_to_param(dist, param_tilde[i])
    # Some treatments 
    NaN2zero!(param[i])
    big_threshold!(param[i], 1e10)
    small_threshold!(param[i], 1e-10)
    return
end

function update_param_tilde!(param_tilde, dist, ω, A, B, score_til, i)
    param_tilde[i + 1] = ω + A*score_til + B*param_tilde[i]
    return 
end