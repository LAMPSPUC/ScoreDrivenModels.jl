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
    biggest_lag = number_of_lags(gas_sarima)
    initial_params = Vector{Vector{T}}(undef, biggest_lag)
    obs_separated = Vector{Vector{T}}(undef, biggest_lag)
    len = length(obs)
    for i in 1:biggest_lag
        idx = collect(i:biggest_lag:len)
        initial_params[i] = vcat(params(fit_mle(D, obs[idx]))...)
    end
    return initial_params
end