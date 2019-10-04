export simulate

function simulate(gas_sarima::GAS_Sarima, n::Int)
    initial_param_tilde = stationary_initial_params(gas_sarima)
    return simulate(gas_sarima, n, initial_param_tilde)
end

function simulate(gas_sarima::GAS_Sarima{D, T}, n::Int, initial_param_tilde::Vector{Vector{T}}) where {D, T}
    # Allocations
    serie = zeros(n)
    param = Vector{Vector{T}}(undef, n)
    param_tilde = Vector{Vector{T}}(undef, n)
    scores_tilde = Vector{Vector{T}}(undef, n)

    biggest_lag = number_of_lags(gas_sarima)

    # initial_values  
    for i in 1:biggest_lag
        param_tilde[i] = initial_param_tilde[i]
        param[i] = param_tilde_to_param(D, initial_param_tilde[i])
        # Sample
        updated_dist = update_dist(D, param_tilde_to_param(D, param_tilde[i]))
        serie[i] = sample_observation(updated_dist)
        scores_tilde[i] = score_tilde(serie[i], D, param[i], param_tilde[i], gas_sarima.scaling)
    end
    
    update_param_tilde!(param_tilde, gas_sarima.ω, gas_sarima.A, gas_sarima.B, scores_tilde, biggest_lag)
    updated_dist = update_dist(D, param_tilde_to_param(D, param_tilde[biggest_lag + 1]))
    serie[biggest_lag + 1] = sample_observation(updated_dist)

    for i in biggest_lag + 1:n-1
        # update step
        univariate_score_driven_update!(param, param_tilde, scores_tilde, serie[i], gas_sarima, i)
        # Sample from the updated distribution
        updated_dist = update_dist(D, param_tilde_to_param(D, param_tilde[i + 1]))
        serie[i + 1] = sample_observation(updated_dist)
    end
    update_param!(param, param_tilde, D, n)

    return serie, param
end