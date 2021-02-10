export simulate, forecast

mutable struct Forecast{T <: AbstractFloat}
    observation_quantiles::Matrix{T}
    observation_forecast::Vector{T}
    observation_scenarios::Matrix{T}
    parameter_quantiles::Array{T, 3}
    parameter_forecast::Matrix{T}
    parameter_scenarios::Array{T, 3}
end

"""
    simulate(series::Vector{T}, gas::Model{D, T}, H::Int, S::Int, kwargs...) where {D, T}

Generate scenarios for the future of a time series by updating the GAS recursion `H` times and taking
a sample of the distribution until it generates `S` scenarios.

By default this method uses the `stationary_initial_params` method to perform the 
score driven recursion. If you estimated the model with a different set of `initial_params`
use them here to maintain the coherence of your estimation.
"""
function simulate(series::Vector{T}, gas::Model{D, T}, H::Int, S::Int;
                    initial_params::Matrix{T} = stationary_initial_params(gas)) where {D, T}
    # Filter params estimated on the time series
    params = score_driven_recursion(gas, series; initial_params = initial_params)
    biggest_lag = number_of_lags(gas)
    n_params = num_params(D)
    params_simulation = params[(end - biggest_lag + 1):end, :]
    # Create scenarios matrix
    observation_scenarios = Matrix{T}(undef, H, S)
    parameter_scenarios = Array{T, 3}(undef, H, n_params, S)
    for scenario in 1:S
        # Notice that we know the parameter time_varying parameters for T + 1 
        # So the last initial_params is already a part of the future simulation
        # And we must take the 
        sim, param = simulate_recursion(gas, H + biggest_lag; initial_params = params_simulation)
        observation_scenarios[:, scenario] = sim[biggest_lag+1:end]
        # The first param is known
        parameter_scenarios[:, :, scenario] = param[biggest_lag+1:end, :]
    end
    return observation_scenarios, parameter_scenarios
end


"""
    forecast(series::Vector{T}, gas::Model{D, T}, H::Int; kwargs...) where {D, T}

Forecast quantiles for future values of a time series by updating the GAS recursion `H` times and 
using Monte Carlo method as in Blasques, Francisco, Siem Jan Koopman,
Katarzyna Lasak and Andre Lucas (2016): "In-Sample Confidence Bounds and Out-of-Sample Forecast
Bands for Time-Varying Parameters in Observation Driven Models", 
International Journal of Forecasting, 32(3), 875-887.

You can pass the desired quantiles as a `Vector{T}`. The default behavior is to 
forecast the median and the `0.025` and `0.975` quantiles.

By default 1000 scenarios are used but you can change this by switching the `S` keyword argument.

By default this method uses the `stationary_initial_params` method to perform the 
score driven recursion. If you estimated the model with a different set of `initial_params`
use them here to maintain the coherence of your estimation.
"""
function forecast(series::Vector{T}, gas::Model{D, T}, H::Int;
                    initial_params::Matrix{T} = stationary_initial_params(gas),
                    quantiles::Vector{T} = T.([0.025, 0.5, 0.975]), S::Int = 10_000) where {D, T}

    observation_scenarios, parameter_scenarios = simulate(series, gas, H, S; 
                                                    initial_params = initial_params)
    parameters_forecast = mean(parameter_scenarios, dims = 3)[:, :, 1]
    observations_forecast = Vector{T}(undef, H)
    for i in 1:H
        dist = update_dist(D, parameters_forecast, i)
        observations_forecast[i] = mean(dist)
    end
    return Forecast(
        get_quantiles(quantiles, observation_scenarios), 
        observations_forecast,
        observation_scenarios,
        get_quantiles(quantiles, parameter_scenarios), 
        parameters_forecast,
        parameter_scenarios
        )
end

function get_quantiles(quantiles_probs::Vector{T}, scenarios::Matrix{T}) where T
    @assert all((quantiles_probs .< 1.0) .& (quantiles_probs .> 0.0))
    unique!(sort!(quantiles_probs))
    quantiles = mapslices(x -> quantile(x, quantiles_probs), scenarios; dims = 2)
    return quantiles
end

function get_quantiles(quantiles_probs::Vector{T}, scenarios::Array{T, 3}) where T
    @assert all((quantiles_probs .< 1.0) .& (quantiles_probs .> 0.0))
    unique!(sort!(quantiles_probs))
    quantiles_per_parameter = Array{T, 3}(undef, 
                                size(scenarios, 1),
                                size(scenarios, 2),
                                length(quantiles_probs)
                                )
    for j in 1:size(scenarios, 2)
        quantiles_per_parameter[:, j, :] = mapslices(x -> quantile(x, quantiles_probs), 
                                                        scenarios[:, j, :]; dims = 2)
    end
    return quantiles_per_parameter
end
