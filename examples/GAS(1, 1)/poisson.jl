push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions

ω = [1.0]
A = [0.9][:, :]
B = [0.9][:, :]
dist = Poisson()
scaling = 0

sd_model = SDModel(ω, A, B, dist, scaling)

initial_values = [15.0]
serie, param = simulate(sd_model, 100, initial_values)

using Plots
plot(vcat(param...), label = "\\lambda")
plot!(serie, label = "y\\_t")

score(1.0, Poisson(3))