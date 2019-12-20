push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
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
#   memory estimate:  235.38 KiB
#   allocs estimate:  18
#   --------------
#   minimum time:     1.084 ms (0.00% GC)
#   median time:      1.130 ms (0.00% GC)
#   mean time:        1.244 ms (2.41% GC)
#   maximum time:     79.034 ms (98.52% GC)
#   --------------
#   samples:          4007
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
#   memory estimate:  466.69 MiB
#   allocs estimate:  40237
#   --------------
#   minimum time:     2.553 s (0.57% GC)
#   median time:      2.655 s (0.59% GC)
#   mean time:        2.655 s (0.59% GC)
#   maximum time:     2.756 s (0.60% GC)
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
#   memory estimate:  353.02 MiB
#   allocs estimate:  30598
#   --------------
#   minimum time:     1.008 s (1.41% GC)
#   median time:      1.265 s (1.36% GC)
#   mean time:        1.281 s (1.35% GC)
#   maximum time:     1.587 s (1.28% GC)
#   --------------
#   samples:          4
#   evals/sample:     1


scaling = 0.5
@benchmark begin
    gas = GAS($p, $q, $LogNormal, $scaling)
    opt_method = ScoreDrivenModels.LBFGS(gas, $num_seeds)
    fit!(gas, $simulation; verbose = $verbose, opt_method = opt_method)
end
# BenchmarkTools.Trial:
#   memory estimate:  8.26 GiB
#   allocs estimate:  159942290
#   --------------
#   minimum time:     7.361 s (18.19% GC)
#   median time:      7.361 s (18.19% GC)
#   mean time:        7.361 s (18.19% GC)
#   maximum time:     7.361 s (18.19% GC)
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
#   memory estimate:  666.69 MiB
#   allocs estimate:  8744603
#   --------------
#   minimum time:     2.908 s (2.08% GC)
#   median time:      3.382 s (2.07% GC)
#   mean time:        3.382 s (2.07% GC)
#   maximum time:     3.855 s (2.07% GC)
#   --------------
#   samples:          2
#   evals/sample:     1