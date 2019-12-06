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
    initial_params = stationary_initial_params(gas)
    return simulate(gas, n, initial_params)
end

function simulate(gas::GAS{D, T}, n::Int, initial_params::Matrix{T}) where {D, T}
    # Allocations
    serie = zeros(n)
    n_params = num_params(D)
    param = Matrix{T}(undef, n, n_params)
    param_tilde = Matrix{T}(undef, n, n_params)
    scores_tilde = Vector{Vector{T}}(undef, n)

    # Auxiliary Allocation
    param_dist = zeros(T, 1, n_params)

    biggest_lag = number_of_lags(gas)

    # initial_values  
    for t in 1:biggest_lag
        for p in 1:n_params
            param[t, p] = initial_params[t, p]
        end
        link!(param_tilde, D, param, t)
        # Sample
        updated_dist = update_dist(D, param, t)
        serie[t] = sample_observation(updated_dist)
        scores_tilde[t] = score_tilde(serie[t], D, param, gas.scaling, t)
    end
    
    update_param_tilde!(param_tilde, gas.ω, gas.A, gas.B, scores_tilde, biggest_lag)
    unlink!(param, D, param_tilde, biggest_lag + 1)
    updated_dist = update_dist(D, param, biggest_lag + 1)
    serie[biggest_lag + 1] = sample_observation(updated_dist)

    for i in biggest_lag + 1:n-1
        # update step
        univariate_score_driven_update!(param, param_tilde, scores_tilde, serie[i], gas, i)
        # Sample from the updated distribution
        unlink!(param, D, param_tilde, i + 1)
        updated_dist = update_dist(D, param, i + 1)
        serie[i + 1] = sample_observation(updated_dist)
    end
    update_param!(param, param_tilde, D, n)

    return serie, param
end