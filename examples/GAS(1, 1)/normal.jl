push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions
using Plots

ω = [1; 0.1]
A = [0.9 0; 0 0.05]
B = [0.9 0; 0 0.05]
dist = Normal()
scaling = 1.0

sd_model = SDModel(ω, A, B, dist, scaling)

initial_values = [10.0, 0.3]
serie, param, param_tilde = simulate(sd_model, 100, initial_values)
param
serie

hcat(param...)'

plot(hcat(param...)', label = ["\\mu" "\\sigma"])
plot!(serie, label = "y\\_t")


# R package
using RCall
# install.packages("GAS")
library(GAS)
A <- diag(c(0.9, 0.05))
B <- diag(c(0.9, 0.05))
kappa <- c(1, 0.1)
Sim <- UniGASSim(fit = NULL, T.sim = 100, kappa = kappa, A = A, B = B, Dist = "beta", ScalingType = "Identity")
plot(Sim)
3
0
