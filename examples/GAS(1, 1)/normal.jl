push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions

ω = [1; 1.0]
A = [0.5 0; 0 0]
B = [0.5 0; 0 0.01]
dist = Normal()
scaling = 1/2

sd_model = SDModel(ω, A, B, dist, scaling)

initial_values = [10.0, 0.3]
serie, param = simulate(sd_model, 100, initial_values)
param
serie

hcat(param...)'

using Plots
plot(hcat(param...)', label = ["\\mu" "\\sigma"])
plot!(serie, label = "y\\_t")