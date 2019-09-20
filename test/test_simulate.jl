push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ScoreDrivenModels
using Distributions
using LinearAlgebra

ω = [1.0; 0.1]
A = convert(Matrix{Float64}, Diagonal([0.0; 0.5]))
B = convert(Matrix{Float64}, Diagonal([0.9; 0.5]))
dist = Normal()
scaling = 0.0

sd_model = SDModel(ω, A, B, dist, scaling)

serie_simulated, param_simulated = simulate(sd_model, 100)


ω = [0.1; 0.1]
A = convert(Matrix{Float64}, Diagonal([0.5; 0.5]))
B = convert(Matrix{Float64}, Diagonal([0.5; 0.5]))
dist = Beta()
scaling = 0.0

sd_model = SDModel(ω, A, B, dist, scaling)

serie_simulated, param_simulated = simulate(sd_model, 1000)

# Nem sempre termina
ω = [NaN; NaN]
A = convert(Matrix{Float64}, Diagonal([NaN; NaN]))
B = convert(Matrix{Float64}, Diagonal([NaN; NaN]))
dist = Beta()
scaling = 0.0

sd_model = SDModel(ω, A, B, dist, scaling)

ScoreDrivenModels.estimate_SDModel!(sd_model, serie_simulated; verbose = 2,
                                    random_seeds_lbfgs = ScoreDrivenModels.RandomSeedsLBFGS(4, ScoreDrivenModels.dimension_unkowns(sd_model)))

sd_model

using Plots

param = score_driven_recursion(sd_model, serie_simulated)
plot(hcat(param_simulated...)'[1:100, :], label = ["real alpha" "real beta"])
plot!(hcat(param...)'[1:100, :], label = ["estimado alpha" "estimado beta"])

include("/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/examples/Extras/useful_plots.jl")
p1 = plot_sdm(serie_simulated, param_simulated, sd_model.dist; quantiles = [0.025; 0.975])
plot_sdm!(p1, serie_simulated, param, sd_model.dist; quantiles = [0.025; 0.975])


