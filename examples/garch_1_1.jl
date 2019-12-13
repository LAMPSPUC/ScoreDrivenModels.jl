push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ARCHModels, ScoreDrivenModels, Distributions

f = fit(GARCH{1, 1}, BG96)
vols = volatilities(f)
me = means(f)
c = coef(f)
loglikelihood(f)

ini = [me[1] vols[1]]

gas = GAS(1, 1, Normal, 1.0, time_varying_params = [2])
# gas.ω[1] = me[1]
# res = estimate!(gas, BG96; opt_method = LBFGS(gas, 10), initial_params = ini)

seed = [c[4]; c[1]; c[3]; c[2] + c[3]]

res = estimate!(gas, BG96; opt_method = LBFGS(gas, [seed], f_tol = 1e-8, g_tol = 1e-6), verbose = 4)
gas
res.llk

# param = score_driven_recursion(gas, BG96; initial_params = ini)
param = score_driven_recursion(gas, BG96)

ScoreDrivenModels.log_likelihood(Normal, BG96, param, 1974)

vols_gas = param[1:end-1, 2]
vols

using Plots
plot([vols_gas vols])


gas = GAS(1, 1, Normal, 1.0, time_varying_params = [2])
gas.ω[1] = c[4]
gas.ω[2] = c[1]
gas.A[1][2, 2] = c[3]
gas.B[1][2, 2] = c[2] + c[3]
param = score_driven_recursion(gas, BG96; initial_params = ini)
param[:, 2]
vols