export score_driven_recursion

"""
score_driven_recursion(sd_model::SDM, observations::Vector{T}) where T

start with the stationary params for a 
"""
function score_driven_recursion(gas_sarima::GAS_Sarima, observations::Vector{T}) where T
    initial_param_tilde = stationary_initial_params(gas_sarima)
    return score_driven_recursion(gas_sarima, observations, initial_param_tilde)
end

function score_driven_recursion(gas_sarima::GAS_Sarima, observations::Vector{T}, initial_param_tilde::Vector{Vector{T}}) where T
    # Allocations
    n = length(observations)
    param = Vector{Vector{T}}(undef, n + 1)
    param_tilde = Vector{Vector{T}}(undef, n + 1)
    scores_tilde = Vector{Vector{T}}(undef, n)

    biggest_lag = length(initial_param_tilde)

    # initial_values  
    for i in 1:biggest_lag
        param_tilde[i] = initial_param_tilde[i]
        param[i] = param_tilde_to_param(gas_sarima.dist, initial_param_tilde[i])
        scores_tilde[i] = score_tilde(observations[i], gas_sarima.dist, param[i], param_tilde[i], gas_sarima.scaling)
    end
    
    update_param_tilde!(param_tilde, gas_sarima.ω, gas_sarima.A, gas_sarima.B, scores_tilde, biggest_lag)

    for i in biggest_lag + 1:n
        univariate_score_driven_update!(param, param_tilde, scores_tilde, observations[i], gas_sarima, i)
    end
    update_param!(param, param_tilde, gas_sarima.dist, n + 1)

    return param
end

function univariate_score_driven_update!(param::Vector{Vector{T}}, param_tilde::Vector{Vector{T}},
                                         scores_tilde::Vector{Vector{T}},
                                         observation::T, sd_model::SDM, i::Int) where T
    # update param 
    update_param!(param, param_tilde, sd_model.dist, i)
    # evaluate score
    scores_tilde[i] = score_tilde(observation, sd_model.dist, param[i], param_tilde[i], sd_model.scaling)
    # update param_tilde
    update_param_tilde!(param_tilde, sd_model.ω, sd_model.A, sd_model.B, scores_tilde, i)
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

function update_param_tilde!(param_tilde::Vector{Vector{T}}, ω::Vector{T}, A::Dict{Int, Matrix{T}}, 
                             B::Dict{Int, Matrix{T}}, score_til::Vector{Vector{T}}, i::Int) where T
    param_tilde[i + 1] = ω
    for (lag, mat) in A
        param_tilde[i + 1] += mat*score_til[i - lag + 1]
    end
    for (lag, mat) in B
        param_tilde[i + 1] += mat*param_tilde[i - lag + 1]
    end
    return 
end