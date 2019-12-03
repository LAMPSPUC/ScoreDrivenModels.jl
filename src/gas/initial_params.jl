export stationary_initial_params_tilde, dynamic_initial_params

"""
    stationary_initial_param_tilde(gas::GAS{D, T}) where {D, T}
"""
function stationary_initial_params_tilde(gas::GAS{D, T}) where {D, T}
    biggest_lag = number_of_lags(gas)
    initial_params_tilde = Vector{Vector{T}}(undef, biggest_lag)
    for i in 1:biggest_lag
        initial_params_tilde[i] = gas.ω./diag(I - gas.B[1])
    end
    return initial_params_tilde
end

"""
    dynamic_initial_params
"""
function dynamic_initial_params(obs::Vector{T}, gas::GAS{D, T}) where {D <: Distribution, T <: AbstractFloat}
    # Take the biggest lag
    biggest_lag = number_of_lags(gas)

    # Allocate memory 
    initial_params = Vector{Vector{T}}(undef, biggest_lag)
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
        sdm_dist = update_dist(D, [params(dist)...])

        initial_params[i] = [params(sdm_dist)...]
    end

    return initial_params
end