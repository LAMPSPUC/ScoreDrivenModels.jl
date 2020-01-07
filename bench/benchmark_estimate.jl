using GAS, Distributions, BenchmarkTools, Random, Test

include("test/utils.jl")

scaling = 0.0
ω = [0.1, 0.1]
A = [0.2 0; 0 0.2]
B = [0.2 0; 0 0.2]
simulation = simulate_GAS_1_1(Beta, scaling, ω, A, B, 1)
gas = GAS.Model(1, 1, Beta, scaling)
gas.ω = ω
gas.A[1] = A
gas.B[1] = B
@benchmark score_driven_recursion($gas, $simulation)
# BenchmarkTools.Trial:
#   memory estimate:  235.38 KiB
#   allocs estimate:  18
#   --------------
#   minimum time:     999.374 μs (0.00% GC)
#   median time:      1.006 ms (0.00% GC)
#   mean time:        1.050 ms (1.74% GC)
#   maximum time:     55.351 ms (98.06% GC)
#   --------------
#   samples:          4756
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
    gas = GAS.Model($p, $q, $Beta, $scaling)
    opt_method = GAS.LBFGS(gas, $num_seeds)
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  516.71 MiB
#   allocs estimate:  50679
#   --------------
#   minimum time:     2.857 s (2.91% GC)
#   median time:      2.927 s (1.71% GC)
#   mean time:        2.927 s (1.71% GC)
#   maximum time:     2.997 s (0.57% GC)
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
    gas = GAS.Model($p, $q, $LogNormal, $scaling)
    opt_method = GAS.LBFGS(gas, $num_seeds)
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  386.33 MiB
#   allocs estimate:  38084
#   --------------
#   minimum time:     1.014 s (1.15% GC)
#   median time:      1.358 s (1.13% GC)
#   mean time:        1.341 s (2.56% GC)
#   maximum time:     1.632 s (1.12% GC)
#   --------------
#   samples:          4
#   evals/sample:     1


scaling = 0.5
@benchmark begin
    gas = GAS.Model($p, $q, $LogNormal, $scaling)
    opt_method = GAS.LBFGS(gas, $num_seeds)
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  1.94 GiB
#   allocs estimate:  54273958
#   --------------
#   minimum time:     4.936 s (3.77% GC)
#   median time:      6.007 s (4.06% GC)
#   mean time:        6.007 s (4.06% GC)
#   maximum time:     7.078 s (4.26% GC)
#   --------------
#   samples:          2
#   evals/sample:     1

scaling = 1.0
@benchmark begin
    gas = GAS.Model($p, $q, $LogNormal, $scaling)
    opt_method = GAS.LBFGS(gas, $num_seeds)
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  750.17 MiB
#   allocs estimate:  9849254
#   --------------
#   minimum time:     3.217 s (1.91% GC)
#   median time:      3.320 s (1.91% GC)
#   mean time:        3.320 s (1.91% GC)
#   maximum time:     3.424 s (1.92% GC)
#   --------------
#   samples:          2
#   evals/sample:     1