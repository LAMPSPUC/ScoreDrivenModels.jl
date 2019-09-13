push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions
using Plots

ω = [0.0]
A = [-0.2][:, :]
B = [0.9][:, :]
dist = Poisson()
scaling = 1.0

sd_model = SDModel(ω, A, B, dist, scaling)

initial_param = [1.0]
serie, param, param_tilde = simulate(sd_model, 100, initial_param)
serie
param
param_tilde
maximum(serie)
maximum(param)

plot(vcat(param...), label = "\\lambda")
plot!(serie, label = "y\\_t")

# R package
using RCall
# install.packages("GAS")
library(GAS)
A <- matrix(c(-0.2))
B <- matrix(c(0.9))
kappa <- c(0.0)
Sim <- UniGASSim(fit = NULL, T.sim = 100, kappa = kappa, A = A, B = B, Dist = "poi", ScalingType = "InvSqrt")
plot(Sim)
obs <- getObs(Sim)
parameters <- getFilteredParameters(Sim)

@rget obs parameters

obs
parameters

println(obs)
println(parameters)
