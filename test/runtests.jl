push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels, Distributions, LinearAlgebra

using Test, Random, HypothesisTests, DelimitedFiles

const SDM = ScoreDrivenModels
cd("test")
include("utils.jl")
include("test_recursion.jl")
# include("test_links.jl")
# include("test_distributions.jl")
# include("test_initial_params.jl")
# include("test_diagnostics.jl")
include("test_estimate.jl")