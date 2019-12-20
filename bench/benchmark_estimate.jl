using ScoreDrivenModels, Distributions, BenchmarkTools, Random, Test

const SDM = ScoreDrivenModels

include("test/utils.jl")

scaling = 0.0
ω = [0.1, 0.1]
A = [0.2 0; 0 0.2]
B = [0.2 0; 0 0.2]
simulation = simulate_GAS_1_1(Beta, scaling, ω, A, B, 1)
p = 1
q = 1
gas = GAS(p, q, Beta, scaling)
gas.ω = ω
gas.A[1] = A
gas.B[1] = B
@benchmark score_driven_recursion($gas, $simulation)
# BenchmarkTools.Trial:
#   memory estimate:  594.09 KiB
#   allocs estimate:  22976
#   --------------
#   minimum time:     1.597 ms (0.00% GC)
#   median time:      1.632 ms (0.00% GC)
#   mean time:        1.756 ms (2.25% GC)
#   maximum time:     52.977 ms (96.78% GC)
#   --------------
#   samples:          2845
#   evals/sample:     1

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
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  1.27 GiB
#   allocs estimate:  51637326
#   --------------
#   minimum time:     4.329 s (2.79% GC)
#   median time:      4.456 s (2.01% GC)
#   mean time:        4.456 s (2.01% GC)
#   maximum time:     4.582 s (1.28% GC)
#   --------------
#   samples:          2
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
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  3.05 GiB
#   allocs estimate:  162735474
#   --------------
#   minimum time:     6.504 s (2.29% GC)
#   median time:      6.504 s (2.29% GC)
#   mean time:        6.504 s (2.29% GC)
#   maximum time:     6.504 s (2.29% GC)
#   --------------
#   samples:          1
#   evals/sample:     1


scaling = 0.5
@benchmark begin
    gas = GAS($p, $q, $LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  3.98 GiB
#   allocs estimate:  187569916
#   --------------
#   minimum time:     9.068 s (4.12% GC)
#   median time:      9.068 s (4.12% GC)
#   mean time:        9.068 s (4.12% GC)
#   maximum time:     9.068 s (4.12% GC)
#   --------------
#   samples:          1
#   evals/sample:     1

scaling = 1.0
@benchmark begin
    gas = GAS($p, $q, $LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  3.44 GiB
#   allocs estimate:  175127573
#   --------------
#   minimum time:     9.172 s (2.29% GC)
#   median time:      9.172 s (2.29% GC)
#   mean time:        9.172 s (2.29% GC)
#   maximum time:     9.172 s (2.29% GC)
#   --------------
#   samples:          1
#   evals/sample:     1