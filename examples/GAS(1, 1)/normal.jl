push!(LOAD_PATH, "/Users/guilhermebodin/Documents/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions
using Plots

ω = [1; 0.1]
A = [0.5 0; 0 0.05]
B = [0.5 0; 0 0.05]
dist = Normal()
scaling = 1/2

sd_model = SDModel(ω, A, B, dist, scaling)

initial_values = [10.0, 0.3]
serie, param, param_tilde = simulate(sd_model, 100, initial_values)
param
serie

hcat(param...)'

plot(hcat(param...)', label = ["\\mu" "\\sigma"])
plot!(serie, label = "y\\_t")