using ScoreDrivenModels, Distributions, LinearAlgebra

using Test, Random, Expectations

const SDM = ScoreDrivenModels

include("test_recursion.jl")
include("test_estimate.jl")
include("test_distributions.jl")
include("test_initial_params.jl")
icnlude("test_diagnostics.jl")