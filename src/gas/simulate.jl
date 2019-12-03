export simulate

function simulate(gas::GAS{D, T}, n::Int, s::Int) where {D, T}
    scenarios_series = Matrix{T}(undef, n, s)

    for i in 1:s
        serie, params = simulate(gas, n)
        scenarios_series[:, i] = serie
    end
    return scenarios_series
end

function simulate(gas::GAS{D, T}, n::Int) where {D, T}
    initial_params_tilde = stationary_initial_params_tilde(gas)

    biggest_lag = number_of_lags(gas)
    initial_params = Vector{Vector{T}}(undef, biggest_lag)
    for i in 1:biggest_lag
        initial_params[i] = unlink(D, initial_params_tilde[i])
    end
    return simulate(gas, n, initial_params)
end

function simulate(gas::GAS{D, T}, n::Int, initial_params::Vector{Vector{T}}) where {D, T}
    # Allocations
    serie = zeros(n)
    param = Vector{Vector{T}}(undef, n)
    param_tilde = Vector{Vector{T}}(undef, n)
    scores_tilde = Vector{Vector{T}}(undef, n)

    biggest_lag = number_of_lags(gas)

    # initial_values  
    for i in 1:biggest_lag
        param[i] = initial_params[i]
        param_tilde[i] = link(D, param[i])
        # Sample
        updated_dist = update_dist(D, unlink(D, param_tilde[i]))
        serie[i] = sample_observation(updated_dist)
        scores_tilde[i] = score_tilde(serie[i], D, param[i], param_tilde[i], gas.scaling)
    end
    
    update_param_tilde!(param_tilde, gas.ω, gas.A, gas.B, scores_tilde, biggest_lag)
    updated_dist = update_dist(D, unlink(D, param_tilde[biggest_lag + 1]))
    serie[biggest_lag + 1] = sample_observation(updated_dist)

    for i in biggest_lag + 1:n-1
        # update step
        univariate_score_driven_update!(param, param_tilde, scores_tilde, serie[i], gas, i)
        # Sample from the updated distribution
        updated_dist = update_dist(D, unlink(D, param_tilde[i + 1]))
        serie[i + 1] = sample_observation(updated_dist)
    end
    update_param!(param, param_tilde, D, n)

    return serie, param
end