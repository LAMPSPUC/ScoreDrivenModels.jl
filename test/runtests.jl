push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels, Distributions, LinearAlgebra

using Test, Random
cd("test")
include("test_recursion.jl")
include("test_estimate.jl")