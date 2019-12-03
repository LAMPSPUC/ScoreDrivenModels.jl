push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
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
#   minimum time:     5.597 s (15.34% GC)
#   median time:      5.597 s (15.34% GC)
#   mean time:        5.597 s (15.34% GC)
#   maximum time:     5.597 s (15.34% GC)
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
    gas = GAS($p, $q, LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    estimate!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  8.63 GiB
#   allocs estimate:  112481074
#   --------------
#   minimum time:     5.500 s (18.56% GC)
#   median time:      5.500 s (18.56% GC)
#   mean time:        5.500 s (18.56% GC)
#   maximum time:     5.500 s (18.56% GC)
#   --------------
#   samples:          1
#   evals/sample:     1


scaling = 0.5
@benchmark begin
    gas = GAS($p, $q, LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    estimate!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  17.39 GiB
#   allocs estimate:  325680000
#   --------------
#   minimum time:     10.204 s (23.11% GC)
#   median time:      10.204 s (23.11% GC)
#   mean time:        10.204 s (23.11% GC)
#   maximum time:     10.204 s (23.11% GC)
#   --------------
#   samples:          1
#   evals/sample:     1

scaling = 1.0
@benchmark begin
    gas = GAS($p, $q, LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    estimate!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  11.99 GiB
#   allocs estimate:  172023821
#   --------------
#   minimum time:     7.312 s (21.09% GC)
#   median time:      7.312 s (21.09% GC)
#   mean time:        7.312 s (21.09% GC)
#   maximum time:     7.312 s (21.09% GC)
#   --------------
#   samples:          1
#   evals/sample:     1