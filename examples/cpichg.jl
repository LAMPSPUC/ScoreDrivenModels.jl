using DelimitedFiles, Random, ScoreDrivenModels, Statistics, Plots

# Set RNG seed to guarantee consistent results
Random.seed!(123)

# Load historical Consumer Price Index data
y = vec(readdlm("../test/data/cpichg.csv"))

# Specify GAS model: a student's t model with location scale transformation
# (see /src/distributions/non_native_dists.jl in the repository)
gas = ScoreDrivenModel(1, 1, TDistLocationScale, 0.0, time_varying_params=[1, 2])

# Fit specified model to historical data
f = fit!(gas, y)

# Next, we show examples using nondefault keyword arguments
# Note that we need to re-define `gas` since its parameters were populated by `fit!`
gas = ScoreDrivenModel(1, 1, TDistLocationScale, 0.0, time_varying_params=[1, 2])

# In this example, we set verbose to 2, prompting the package to output optimization results
f = fit!(gas, y; verbose=2)

# Re-define `gas` once again for another example
gas = ScoreDrivenModel(1, 1, TDistLocationScale, 0.0, time_varying_params=[1, 2])

# In this example, we set the optimization method to LBFGS with 5 initial points
f = fit!(gas, y; opt_method=LBFGS(gas, 5))

# Print estimation statistics
results(f)
plot(f)

# Perform forecast via simulations for 12 time periods ahead
forec = forecast(y, gas, 12)

# We can access the parameter forecasts in `forec.parameter_forecasts`
forec.parameter_forecast

# Similarly, we can access the simulated observation scenarios
forec.observation_scenarios

gas_t = ScoreDrivenModel(1, 1, TDistLocationScale, 1.0; time_varying_params = [1])
gas_t_1_2 = ScoreDrivenModel(1, 2, TDistLocationScale, 1.0; time_varying_params = [1])
steps_ahead = 50
first_idx = 150
b_t = cross_validation(gas_t, y, steps_ahead, first_idx)
b_t_1_2 = cross_validation(gas_t_1_2, y, steps_ahead, first_idx)
plot(b_t, "GAS(1, 1) Student t")
plot!(b_t_1_2, "GAS(1, 2) Student t"; legend = :topleft)
