push!(LOAD_PATH, "/Users/guilhermebodin/Documents/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions
using Plots

ω = [0.01]
A = [0.3][:, :]
B = [0.4][:, :]
dist = Poisson()
scaling = 0.5

sd_model = SDModel(ω, A, B, dist, scaling)

initial_param = [15.0]
serie, param, param_tilde = simulate(sd_model, 100, initial_param)
serie
param
param_tilde
maximum(serie)
maximum(param)

plot(vcat(param...), label = "\\lambda")
plot!(serie, label = "y\\_t")

# R package
