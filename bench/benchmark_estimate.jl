push!(LOAD_PATH, "/Users/guilhermebodin/Documents/ScoreDrivenModels.jl/src")
using ScoreDrivenModels, Distributions, BenchmarkTools, Random, Test

include("test/utils.jl")

scaling = 0.0
ω = [0.1, 0.1]
A = [0.2 0; 0 0.2]
B = [0.2 0; 0 0.2]
simulation = simulate_GAS_1_1(Beta, scaling, ω, A, B, 1)
verbose = 0
p = 1
q = 1
num_seeds = 3
@benchmark begin
    gas = GAS($p, $q, $Beta, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    estimate!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  7.33 GiB
#   allocs estimate:  95576498
#   --------------
#   minimum time:     5.632 s (15.93% GC)
#   median time:      5.632 s (15.93% GC)
#   mean time:        5.632 s (15.93% GC)
#   maximum time:     5.632 s (15.93% GC)
#   --------------
#   samples:          1
#   evals/sample:     1

scaling = 0.0
ω = [0.5, 0.5]
A = [0.1 0; 0 0.1]
B = [0.1 0; 0 0.1]
simulation = simulate_GAS_1_1(LogNormal, scaling, ω, A, B, 4)
verbose = 0
p = 1
q = 1
num_seeds = 3
@benchmark begin
    gas = GAS($p, $q, $LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    estimate!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial: 
#   memory estimate:  1.24 GiB
#   allocs estimate:  9806219
#   --------------
#   minimum time:     3.769 s (2.16% GC)
#   median time:      3.957 s (3.17% GC)
#   mean time:        3.957 s (3.17% GC)
#   maximum time:     4.146 s (4.10% GC)
#   --------------
#   samples:          2
#   evals/sample:     1


scaling = 0.5
@benchmark begin
    gas = GAS($p, $q, $LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    estimate!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  17.39 GiB
#   allocs estimate:  325680001
#   --------------
#   minimum time:     9.760 s (23.28% GC)
#   median time:      9.760 s (23.28% GC)
#   mean time:        9.760 s (23.28% GC)
#   maximum time:     9.760 s (23.28% GC)
#   --------------
#   samples:          1
#   evals/sample:     1

scaling = 1.0
@benchmark begin
    gas = GAS($p, $q, $LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    estimate!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  11.99 GiB
#   allocs estimate:  172023821
#   --------------
#   minimum time:     8.502 s (20.62% GC)
#   median time:      8.502 s (20.62% GC)
#   mean time:        8.502 s (20.62% GC)
#   maximum time:     8.502 s (20.62% GC)
#   --------------
#   samples:          1
#   evals/sample:     1