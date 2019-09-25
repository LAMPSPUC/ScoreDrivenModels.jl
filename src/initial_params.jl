"""
    stationary_initial_params(gas_sarima::GAS_Sarima)
"""
function stationary_initial_params(gas_sarima::GAS_Sarima)
    biggest_lag = max(length(gas_sarima.A), length(gas_sarima.B))
    initial_params = Vector{Vector{Float64}}(undef, biggest_lag)
    for i in 1:biggest_lag
        initial_params[i] = gas_sarima.ω./diag(I - gas_sarima.B[1])
    end
    return initial_params
end

# function dynamic_initial_params(obs::Vector{T}, dist::Distribution, biggest_lag::Int) where T
    
# end