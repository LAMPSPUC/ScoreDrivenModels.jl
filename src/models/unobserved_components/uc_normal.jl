include("/Users/guilhermebodin/Documents/ScoreDrivenModels.jl/src/models/unobserved_components/common.jl")

mutable struct UnobservedComponentsNormal
    y::Vector{Float64}
    parameter_dynamics::Vector{UCParameterDynamics}
end