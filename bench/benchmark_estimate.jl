using ScoreDrivenModels, Distributions, BenchmarkTools, Random

function simulate_GAS_1_1(D::Type{<:Distribution}, scaling::Real, ω::Vector{T}, A::Matrix{T}, 
                            B::Matrix{T}, seed::Int) where T
    Random.seed!(seed)
    gas = GAS(1, 1, D, scaling)

    gas.ω = ω
    gas.A[1] = A
    gas.B[1] = B
    series, param = simulate(gas, 5000)

    return series
end

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
#   memory estimate:  7.78 GiB
#   allocs estimate:  101428072
#   --------------
#   minimum time:     7.013 s (8.76% GC)
#   median time:      7.013 s (8.76% GC)
#   mean time:        7.013 s (8.76% GC)
#   maximum time:     7.013 s (8.76% GC)
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
#   minimum time:     6.900 s (10.20% GC)
#   median time:      6.900 s (10.20% GC)
#   mean time:        6.900 s (10.20% GC)
#   maximum time:     6.900 s (10.20% GC)
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
#   minimum time:     11.689 s (13.11% GC)
#   median time:      11.689 s (13.11% GC)
#   mean time:        11.689 s (13.11% GC)
#   maximum time:     11.689 s (13.11% GC)
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
#   allocs estimate:  172023822
#   --------------
#   minimum time:     8.737 s (11.14% GC)
#   median time:      8.737 s (11.14% GC)
#   mean time:        8.737 s (11.14% GC)
#   maximum time:     8.737 s (11.14% GC)
#   --------------
#   samples:          1
#   evals/sample:     1