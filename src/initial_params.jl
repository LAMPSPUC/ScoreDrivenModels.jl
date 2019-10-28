export stationary_initial_params, dynamic_initial_params

"""
    stationary_initial_params(gas_sarima::GAS_Sarima)
"""
function stationary_initial_params(gas_sarima::GAS_Sarima)
    biggest_lag = number_of_lags(gas_sarima)
    initial_params = Vector{Vector{Float64}}(undef, biggest_lag)
    for i in 1:biggest_lag
        initial_params[i] = gas_sarima.ω./diag(I - gas_sarima.B[1])
    end
    return initial_params
end

"""
    dynamic_initial_params
"""
function dynamic_initial_params(obs::Vector{T}, gas_sarima::GAS_Sarima{D, T}) where {D <: Distribution, T <: AbstractFloat}
    # Take the biggest lag
    biggest_lag = number_of_lags(gas_sarima)

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