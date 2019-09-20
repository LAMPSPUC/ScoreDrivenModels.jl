push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels, Distributions, LinearAlgebra

using Test, Random


gas_sarima = GAS_Sarima(1, 1, Beta(), 0.0)
vect = 0.1*ones(length(params(Beta())))
gas_sarima.ω = vect
gas_sarima.A[1] = convert(Matrix{Float64}, Diagonal(5*vect))
gas_sarima.B[1] = convert(Matrix{Float64}, Diagonal(5*vect))  
gas_sarima
simulation, param_simulated = simulate(gas_sarima, 1000)

gas_sarima = GAS_Sarima(1, 1, Normal(), 0.0)
estimate_GAS_Sarima!(gas_sarima, simulation; verbose = 0,
                     random_seeds_lbfgs = ScoreDrivenModels.RandomSeedsLBFGS(5, ScoreDrivenModels.dim_unknowns(gas_sarima)))

gas_sarima
param = score_driven_recursion(gas_sarima, simulation)
param_simulated

include("test_recursion.jl")
include("test_estimate.jl")