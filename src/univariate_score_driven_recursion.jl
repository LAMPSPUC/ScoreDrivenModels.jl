export score_driven_recursion, fitted_mean

"""
score_driven_recursion(sd_model::SDM, observations::Vector{T}) where T

start with the stationary params for a 
"""
function score_driven_recursion(gas_sarima::GAS_Sarima, observations::Vector{T}) where T
    initial_param_tilde = stationary_initial_params(gas_sarima)
    return score_driven_recursion(gas_sarima, observations, initial_param_tilde)
end

function score_driven_recursion(gas_sarima::GAS_Sarima{D, T}, observations::Vector{T}, initial_param_tilde::Vector{Vector{T}}) where {D, T}
    # Allocations
    n = length(observations)
    param = Vector{Vector{T}}(undef, n + 1)
    param_tilde = Vector{Vector{T}}(undef, n + 1)
    scores_tilde = Vector{Vector{T}}(undef, n)

    biggest_lag = length(initial_param_tilde)

    # initial_values  
    for i in 1:biggest_lag
        param_tilde[i] = initial_param_tilde[i]
        param[i] = param_tilde_to_param(D, initial_param_tilde[i])
        scores_tilde[i] = score_tilde(observations[i], D, param[i], param_tilde[i], gas_sarima.scaling)
    end
    
    update_param_tilde!(param_tilde, gas_sarima.ω, gas_sarima.A, gas_sarima.B, scores_tilde, biggest_lag)

    for i in biggest_lag + 1:n
        univariate_score_driven_update!(param, param_tilde, scores_tilde, observations[i], gas_sarima, i)
    end
    update_param!(param, param_tilde, D, n + 1)

    return param
end

function univariate_score_driven_update!(param::Vector{Vector{T}}, param_tilde::Vector{Vector{T}},
                                         scores_tilde::Vector{Vector{T}},
                                         observation::T, gas_sarima::GAS_Sarima{D, T}, i::Int) where {D <: Distribution, T <: AbstractFloat}
    # update param 
    update_param!(param, param_tilde, D, i)
    # evaluate score
    scores_tilde[i] = score_tilde(observation, D, param[i], param_tilde[i], gas_sarima.scaling)
    # update param_tilde
    update_param_tilde!(param_tilde, gas_sarima.ω, gas_sarima.A, gas_sarima.B, scores_tilde, i)
    return 
end

function update_param!(param::Vector{Vector{T}}, param_tilde::Vector{Vector{T}}, D::Type{<:Distribution}, i::Int) where T
    param[i] = param_tilde_to_param(D, param_tilde[i])
    # Some treatments 
    NaN2zero!(param[i])
    big_threshold!(param[i], 1e10)
    small_threshold!(param[i], 1e-10)
    return
end

function update_param_tilde!(param_tilde::Vector{Vector{T}}, ω::Vector{T}, A::Dict{Int, Matrix{T}}, 
                             B::Dict{Int, Matrix{T}}, scores_tilde::Vector{Vector{T}}, i::Int) where T
    param_tilde[i + 1] = copy(ω)
    for (lag, mat) in A
        param_tilde[i + 1] .+= mat*scores_tilde[i - lag + 1]
    end
    for (lag, mat) in B
        param_tilde[i + 1] .+= mat*param_tilde[i - lag + 1]
    end
    return 
end

"""
    fitted_mean(gas_sarima::GAS_Sarima{D, T}, observations::Vector{T}, initial_params::Vector{T}) where {D, T}

return the fitted mean.. #TODO
"""
function fitted_mean(gas_sarima::GAS_Sarima{D, T}, observations::Vector{T}, initial_params::Vector{Vector{T}}) where {D, T}
    params_fitted = score_driven_recursion(gas_sarima, observations, initial_params)
    n = length(params_fitted)
    fitted_mean = Vector{T}(undef, n)

    for (i, param) in enumerate(params_fitted)
        fitted_mean[i] = mean(D(param...))
    end
    
    return fitted_mean
end