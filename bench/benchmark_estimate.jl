using ScoreDrivenModels, Distributions, BenchmarkTools, Random, Test

include("test/utils.jl")

scaling = 0.0
ω = [0.1, 0.1]
A = [0.2 0; 0 0.2]
B = [0.2 0; 0 0.2]
simulation = simulate_GAS_1_1(LogNormal, scaling, ω, A, B, 1)
gas = ScoreDrivenModel(1, 1, LogNormal, scaling)
gas.ω = ω
gas.A[1] = A
gas.B[1] = B
@benchmark score_driven_recursion($gas, $simulation)
# BenchmarkTools.Trial:
#   memory estimate:  235.38 KiB
#   allocs estimate:  18
#   --------------
#   minimum time:     457.305 μs (0.00% GC)
#   median time:      467.132 μs (0.00% GC)
#   mean time:        485.643 μs (2.39% GC)
#   maximum time:     52.405 ms (99.08% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1

scaling = 0.5
ω = [0.1, 0.1]
A = [0.2 0; 0 0.2]
B = [0.2 0; 0 0.2]
simulation = simulate_GAS_1_1(LogNormal, scaling, ω, A, B, 1)
gas = ScoreDrivenModel(1, 1, LogNormal, scaling)
gas.ω = ω
gas.A[1] = A
gas.B[1] = B
@benchmark score_driven_recursion($gas, $simulation)
# BenchmarkTools.Trial:
#   memory estimate:  938.50 KiB
#   allocs estimate:  25018
#   --------------
#   minimum time:     2.018 ms (0.00% GC)
#   median time:      2.045 ms (0.00% GC)
#   mean time:        2.193 ms (3.19% GC)
#   maximum time:     5.779 ms (55.53% GC)
#   --------------
#   samples:          2278
#   evals/sample:     1

scaling = 1.0
ω = [0.1, 0.1]
A = [0.2 0; 0 0.2]
B = [0.2 0; 0 0.2]
simulation = simulate_GAS_1_1(LogNormal, scaling, ω, A, B, 1)
gas = ScoreDrivenModel(1, 1, LogNormal, scaling)
gas.ω = ω
gas.A[1] = A
gas.B[1] = B
@benchmark score_driven_recursion($gas, $simulation)
# BenchmarkTools.Trial:
#   memory estimate:  391.63 KiB
#   allocs estimate:  5018
#   --------------
#   minimum time:     1.450 ms (0.00% GC)
#   median time:      1.474 ms (0.00% GC)
#   mean time:        1.538 ms (1.69% GC)
#   maximum time:     5.374 ms (58.97% GC)
#   --------------
#   samples:          3247
#   evals/sample:     1
