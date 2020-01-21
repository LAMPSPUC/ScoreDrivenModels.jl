export simulate, forecast

"""
    simulate(serie::Vector{T}, gas::Model{D, T}, N::Int, S::Int, kwargs...) where {D, T}

Generate scenarios for the future of a time series by updating the GAS recursion `N` times and taking 
a sample of the distribution until generate `S` scenarios.
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
    forecast(serie::Vector{T}, gas::Model{D, T}, N::Int; kwargs...) where {D, T}

Forecast future values of a time series by updating the GAS recursion `N` times and 
taking the mean of the distribution at each time. You can pass the desired confidence interval 
as a `Vector{T}`. The forecast will be the first column and the confidence intervals are the remaining
columns.

The forecast is build using Monte Carlo method as in Blasques, Francisco, Siem Jan Koopman, 
Katarzyna Lasak and Andre Lucas (2016): "In-Sample Confidence Bounds and Out-of-Sample Forecast 
Bands for Time-Varying Parameters in Observation Driven Models", International Journal of Forecasting, 32(3), 875-887. 

By default 1000 scenarios are used but one can change by switching the `S` keyword argument.
"""
function forecast(serie::Vector{T}, gas::Model{D, T}, N::Int;
                    initial_params::Matrix{T} = stationary_initial_params(gas),
                    ci::Vector{T} = T.([0.95]), S::Int = 1000) where {D, T}

    scenarios = simulate(serie, gas, N, S; initial_params = initial_params)
    
    quantiles = get_quantiles(ci)

    forec = Matrix{T}(undef, N, length(quantiles) + 1)
    
    forec[:, 1] = mean(scenarios, dims = 2)
    for t in 1:N
        for (i, q) in enumerate(quantiles)
            forec[t, i + 1] = quantile(scenarios[t, :], q)
        end
    end
    
    return forec
end

function get_quantiles(ci::Vector{T}) where T
    @assert all((ci .< 1.0) .& (ci .> 0.0))
    quantiles = zeros(2 * length(ci))
    i = 1
    for v in ci
        quantiles[i] = 0.5 + v/2
        i += 1
        quantiles[i] = 0.5 - v/2
        i += 1
    end
    return quantiles
end