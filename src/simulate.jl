export simulate, forecast_ci, forecast

"""
    simulate(serie::Vector{T}, gas::Model{D, T}, N::Int, S::Int, kwargs...) where {D, T}

Generate scenarios for the future of a time series by updating the GAS recursion `N` times and taking
a sample of the distribution until it generates `S` scenarios.

By default this method uses the `stationary_initial_params` method to perform the 
score driven recursion. If you estimated the model with a different set of `initial_params`
pass it here to maintain the coherence of your estimation.
"""
function simulate(serie::Vector{T}, gas::Model{D, T}, N::Int, S::Int;
                    initial_params::Matrix{T} = stationary_initial_params(gas)) where {D, T}
    # Filter params estimated on the time series
    params = score_driven_recursion(gas, serie; initial_params = initial_params)

    biggest_lag = number_of_lags(gas)

    params_simulation = params[(end - biggest_lag):(end - 1), :]
    # Create scenarios matrix
    scenarios = Matrix{T}(undef, N, S)
    for scenario in 1:S
        sim, param = simulate_recursion(gas, N + biggest_lag + 1; initial_params = params_simulation)
        scenarios[:, scenario] = sim[biggest_lag + 2:end]
    end

    return scenarios
end


"""
    forecast_ci(serie::Vector{T}, gas::Model{D, T}, N::Int; kwargs...) where {D, T}

Forecast confidence intervals future values of a time series by updating the GAS recursion `N` times and 
using Monte Carlo method as in Blasques, Francisco, Siem Jan Koopman,
Katarzyna Lasak and Andre Lucas (2016): "In-Sample Confidence Bounds and Out-of-Sample Forecast
Bands for Time-Varying Parameters in Observation Driven Models", 
International Journal of Forecasting, 32(3), 875-887.
 
You can pass the desired confidence interval as a `Vector{T}`. The default value is to 
forecast in the median, `0.025` and `0.975` quantiles.

By default 1000 scenarios are used but one can change by switching the `S` keyword argument.

By default this method uses the `stationary_initial_params` method to perform the 
score driven recursion. If you estimated the model with a different set of `initial_params`
pass it here to maintain the coherence of your estimation.
"""
function forecast_ci(serie::Vector{T}, gas::Model{D, T}, N::Int;
                      initial_params::Matrix{T} = stationary_initial_params(gas),
                      ci::Vector{T} = T.([0.5, 0.95]), S::Int = 1000) where {D, T}

    scenarios = simulate(serie, gas, N, S; initial_params = initial_params)

    return get_quantiles(ci, scenarios)
end

"""
    forecast(serie::Vector{T}, gas::Model{D, T}, N::Int; kwargs...) where {D, T}

Forecast the expected future values of a time series by updating the GAS recursion `N` times and 
using Monte Carlo method as in Blasques, Francisco, Siem Jan Koopman,
Katarzyna Lasak and Andre Lucas (2016): "In-Sample Confidence Bounds and Out-of-Sample Forecast
Bands for Time-Varying Parameters in Observation Driven Models", 
International Journal of Forecasting, 32(3), 875-887.

By default 1000 scenarios are used but one can change by switching the `S` keyword argument.

By default this method uses the `stationary_initial_params` method to perform the 
score driven recursion. If you estimated the model with a different set of `initial_params`
pass it here to maintain the coherence of your estimation.
"""
function forecast(serie::Vector{T}, gas::Model{D, T}, N::Int;
                    initial_params::Matrix{T} = stationary_initial_params(gas),
                    S::Int = 1000) where {D, T}

    scenarios = simulate(serie, gas, N, S; initial_params = initial_params)

    return mean(scenarios, dims = 2)
end

function get_quantiles(ci::Vector{T}, scenarios::Matrix{T}) where T
    @assert all((ci .< 1.0) .& (ci .> 0.0))
    cis = unique(sort([ci; 1 .- ci]))
    quantiles = mapslices(x -> quantile(x, cis), scenarios; dims = 2)
    return quantiles
end
