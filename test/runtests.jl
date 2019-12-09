using ScoreDrivenModels, Distributions, LinearAlgebra

using Test, Random, HypothesisTests

const SDM = ScoreDrivenModels

include("utils.jl")
include("test_links.jl")
include("test_distributions.jl")

# GAS
include("test_recursion.jl")
include("test_initial_params.jl")
include("test_diagnostics.jl")
include("test_estimate.jl")