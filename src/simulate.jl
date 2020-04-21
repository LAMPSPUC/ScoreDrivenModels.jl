export simulate, forecast_quantiles

mutable struct SDMForecast{T <: AbstractFloat}
    quantiles::Matrix{T}
    scenarios::Matrix{T}
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
    params_simulation = params[(end - biggest_lag):(end - 1), :]
    # Create scenarios matrix
    scenarios = Matrix{T}(undef, H, S)
    for scenario in 1:S
        sim, param = simulate_recursion(gas, H + biggest_lag + 1; initial_params = params_simulation)
        scenarios[:, scenario] = sim[biggest_lag + 2:end]
    end
    return scenarios
end


"""
    forecast_quantiles(series::Vector{T}, gas::Model{D, T}, H::Int; kwargs...) where {D, T}

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
function forecast_quantiles(series::Vector{T}, gas::Model{D, T}, H::Int;
                      initial_params::Matrix{T} = stationary_initial_params(gas),
                      quantiles::Vector{T} = T.([0.025, 0.5, 0.975]), S::Int = 10_000) where {D, T}

    scenarios = simulate(series, gas, H, S; initial_params = initial_params)
    return SDMForecast(get_quantiles(quantiles, scenarios), scenarios)
end

function get_quantiles(quantile_probs::Vector{T}, scenarios::Matrix{T}) where T
    @assert all((quantile_probs .< 1.0) .& (quantile_probs .> 0.0))
    unique!(sort!(quantile_probs))
    quantiles = mapslices(x -> quantile(x, quantile_probs), scenarios; dims = 2)
    return quantiles
end
