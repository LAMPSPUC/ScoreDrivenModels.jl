export simulate, forecast

"""
    simulate(serie::Vector{T}, gas::GAS{D, T}, N::Int, S::Int, kwargs...) where {D, T}

Generate scenarios for the future of a time series by updating the GAS recursion `N` times and taking 
a sample of the distribution until generate `S` scenarios.
"""
function simulate(serie::Vector{T}, gas::GAS{D, T}, N::Int, S::Int;
                    initial_params::Matrix{T} = stationary_initial_params(gas)) where {D, T}
    # Filter params estimated on the time series
    params = score_driven_recursion(gas, serie; initial_params = initial_params)

    biggest_lag = number_of_lags(gas)

    params_simulation = params[(end - biggest_lag):(end - 1), :]
    # Create scenarios matrix
    scenarios = Matrix{T}(undef, N, S)
    for scenario in 1:S
        sim, param = simulate_recursion(gas, N + biggest_lag; initial_params = params_simulation)
        scenarios[:, scenario] = sim[biggest_lag + 1:end]
    end

    return scenarios
end


"""
    forecast(serie::Vector{T}, gas::GAS{D, T}, N::Int; kwargs...) where {D, T}

Forecast future values of a time series by updating the GAS recursion `N` times and 
taking the mean of the distribution at each time.
"""
function forecast(serie::Vector{T}, gas::GAS{D, T}, N::Int;
                    initial_params::Matrix{T} = stationary_initial_params(gas)) where {D, T}
    # Filter params estimated on the time series
    params = score_driven_recursion(gas, serie; initial_params = initial_params)

    biggest_lag = number_of_lags(gas)

    params_simulation = params[(end - biggest_lag):(end - 1), :]
    # Create scenarios matrix
    forec = Vector{T}(undef, N)
    sim, param = simulate_recursion(gas, N + biggest_lag; initial_params = params_simulation, update = mean)
    forec = sim[biggest_lag + 1:end]

    return forec
end