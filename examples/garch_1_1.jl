push!(LOAD_PATH, "/home/guilhermebodin/Documents/Github/ScoreDrivenModels.jl/src")
using ARCHModels, ScoreDrivenModels, Distributions

f = fit(GARCH{1, 1}, BG96)
vols = volatilities(f)
me = means(f)
c = coef(f)

ini = [me[1] vols[1]]

gas = GAS(1, 1, Normal, 1.0, time_varying_params = [2])
# gas.ω[1] = me[1]
# res = estimate!(gas, BG96; opt_method = LBFGS(gas, 10), initial_params = ini)
res = estimate!(gas, BG96; opt_method = LBFGS(gas, 10))
gas

# param = score_driven_recursion(gas, BG96; initial_params = ini)
param = score_driven_recursion(gas, BG96)

vols_gas = param[1:end-1, 2]
vols

using Plots
plot([vols_gas vols])