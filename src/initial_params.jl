export stationary_initial_params_tilde, stationary_initial_params, dynamic_initial_params

"""
    stationary_initial_param_tilde(gas::Model{D, T}) where {D, T}

#TODO
"""
function stationary_initial_params_tilde(gas::Model{D, T}) where {D, T}
    biggest_lag = number_of_lags(gas)
    n_params = num_params(D)
    initial_params_tilde = Matrix{T}(undef, biggest_lag, n_params)
    for t in 1:biggest_lag, p in 1:n_params
        # ω/(1 - sum(B[p, p]))
        initial_params_tilde[t, p] = gas.ω[p]/(1 - sum(v[p, p] for (k, v) in gas.B))
    end
    return initial_params_tilde
end

"""
    stationary_initial_params(gas::Model{D, T}) where {D, T}

#TODO
"""
function stationary_initial_params(gas::Model{D, T}) where {D, T}
    biggest_lag = number_of_lags(gas)
    n_params = num_params(D)
    initial_params_tilde = Matrix{T}(undef, biggest_lag, n_params)
    initial_params = Matrix{T}(undef, biggest_lag, n_params)
    for t in 1:biggest_lag, p in 1:n_params
        # ω/(1 - sum(B[p, p]))
        initial_params_tilde[t, p] = gas.ω[p]/(1 - sum(v[p, p] for (k, v) in gas.B))
        unlink!(initial_params, D, initial_params_tilde, t)
    end
    return initial_params
end

"""
    dynamic_initial_params

    #TODO This is an heuristic! only seen in some phd thesis.
"""
function dynamic_initial_params(obs::Vector{T}, gas::Model{D, T}) where {D, T}
    # Take the biggest lag
    biggest_lag = number_of_lags(gas)

    # Allocate memory 
    initial_params = Matrix{T}(undef, biggest_lag, num_params(D))
    obs_separated = Vector{Vector{T}}(undef, biggest_lag)

    # Loop to fit mle in every component of seasonality
    len = length(obs)
    for i in 1:biggest_lag
        # Indexes of a component of seasonality i.e. of january in a annual seasonality
        idx = collect(i:biggest_lag:len)
        # Fit MLE in the observations
        dist = fit_mle(D, obs[idx])
        # Adequate to the ScoreDrivenModels standard
        # In Distributions Normal is \mu and \sigma
        # In ScoreDrivenModels Normal is \mu and \sigma^2
        sdm_dist = update_dist(D, permutedims([params_sdm(dist)...]), 1)

        initial_params[i, :] = [params_sdm(sdm_dist)...]
    end

    return initial_params
end