export score_driven_recursion

"""
score_driven_recursion(sd_model::SDM, observations::Vector{T}) where T

start with the stationary params for a 
"""
function score_driven_recursion(sd_model::SDM, observations::Vector{T}) where T
    initial_param_tilde = stationary_initial_params(sd_model)
    return score_driven_recursion(sd_model, observations, param_tilde_to_param(sd_model.dist, initial_param_tilde))
end

function score_driven_recursion(sd_model::SDM, observations::Vector{T}, initial_param::Vector{Vector{T}}) where T
    # Allocations
    n = length(observations)
    param = Vector{Vector{T}}(undef, n + 1)
    param_tilde = Vector{Vector{T}}(undef, n + 1)
    score_tildes = Vector{Vector{T}}(undef, n)
    # initial_values 
    for i in 1:length(initial_param)
        param_tilde[i] = param_to_param_tilde(sd_model.dist, initial_param[i])
    end

    for i in 1:n
        univariate_score_driven_update!(param, param_tilde, observations[i], sd_model, i)
    end
    update_param!(param, param_tilde, sd_model.dist, n + 1)

    return param
end

function univariate_score_driven_update!(param::Vector{Vector{T}}, param_tilde::Vector{Vector{T}}, 
                                         observation::T, sd_model::SDM, i::Int) where T
    # update param 
    update_param!(param, param_tilde, sd_model.dist, i)
    # evaluate score
    score_til = score_tilde(observation, sd_model.dist, param[i], param_tilde[i], sd_model.scaling)
    # update param_tilde
    update_param_tilde!(param_tilde, sd_model.ω, sd_model.A, sd_model.B, score_til, i)
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

function update_param_tilde!(param_tilde::Vector{Vector{T}}, ω::Vector{T}, A, B, score_til, i) where T
    param_tilde[i + 1] = ω + A*score_til + B*param_tilde[i]
    return 
end