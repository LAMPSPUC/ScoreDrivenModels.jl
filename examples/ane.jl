using Dates, DelimitedFiles, Plots, Random, ScoreDrivenModels

# Define dates and load historical Affluent Natural Energy data
dates = collect(Date(1961):Month(1):Date(2000, 12))
y = vec(readdlm("../test/data/ane_northeastern.csv"))
y_train = y[1:400]
y_test = y[401:460]

# Set RNG seed to guarantee consistent results
Random.seed!(123)

# Specify GAS model: a lognormal model with time-varying μ, constant σ, and lags 4 and 12
gas = Model(4, 12, LogNormal, 1.0; time_varying_params=[1])

# Obtain initial parameters to start the GAS recursion
initial_params = dynamic_initial_params(y_train, gas)

# Fit specified model to historical data using initial parameters
f = ScoreDrivenModels.fit!(gas, y_train; initial_params=initial_params)

# Print estimation statistics
results(f)
plot(f)

# Simulate 1000 future scenarios and obtain the 5% and 95% quantiles in each time period
forec = forecast(y_train, gas, 60; S=1000, initial_params=initial_params)

# Plot results
plot(dates[401:460], forec.observation_scenarios, color="grey", width=0.05, label="", ylims=(0, 70))
plot!(dates[360:460], y[360:460], label="ANE", color="black", xlabel="Months", ylabel="GWmed", legend=:topright)
plot!(dates[401:460], forec.observation_quantiles, label=["Quantiles" "" ""], color="red", line=:dash)
