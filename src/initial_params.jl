"""
    AbstractInitializationMethod

Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractInitializationMethod end

"""
    stationary_initial_params(sd_model::SDModel)
"""
function stationary_initial_params(sd_model::SDModel)
    return sd_model.ω./diag(I - sd_model.B)
end