"""
    AbstractInitializationMethod

Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractInitializationMethod end

"""
    stationary_initial_params(sd_model::GAS_Sarima)
"""
function stationary_initial_params(gas_sarima::GAS_Sarima)
    return gas_sarima.ω./diag(I - sum_lags_matrices(gas_sarima.B))
end