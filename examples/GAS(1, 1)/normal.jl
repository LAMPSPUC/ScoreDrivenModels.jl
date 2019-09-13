push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions
using Plots
using LinearAlgebra

ω = [1.0; 0.1]
A = Diagonal([0.0; 0.5])
B = Diagonal([0.9; 0.5])
dist = Normal()
scaling = 0.0

sd_model = SDModel(ω, A, B, dist, scaling)

serie, param, param_tilde = simulate(sd_model, 100)
param
serie

hcat(param...)'

param[1]
param_tilde[1]

param[47]
param_tilde[47]

plot(hcat(param...)', label = ["\\mu" "\\sigma"])
plot!(serie, label = "y\\_t")


# R package
using RCall
# install.packages("GAS")
library(GAS)
A <- diag(c(0.0, 0.5))
B <- diag(c(0.0, 0.5))
kappa <- c(1.0, 0.1)
Sim <- UniGASSim(fit = NULL, T.sim = 100, kappa = kappa, A = A, B = B, Dist = "norm", ScalingType = "Identity")
plot(Sim)
obs <- getObs(Sim)
parameters <- getFilteredParameters(Sim)

@rget obs parameters
