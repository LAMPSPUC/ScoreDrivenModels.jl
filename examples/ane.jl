using ScoreDrivenModels, Plots

# Convert data to vector
y = Vector{Float64}(vec(readdlm("./test/data/ena_northeastern.csv")'))
y_train = y[1:400]
y_test = y[401:end]

# Specify model: here we use lag 1 for trend characterization and lag 12 for seasonality characterization
gas = Model([1, 2, 11, 12], [1, 2, 11, 12], LogNormal, 0.0; time_varying_params = [1])

# Define initial_params with
initial_params = dynamic_initial_params(y_train, gas)

# Estimate the model via MLE
f = fit!(gas, y_train; initial_params = initial_params)

# Compare observations and in-sample estimates
plot(y_train, label = "In-sample ANE")

# Forecasts with 95% confidence interval
forecast = forecast_quantiles(y_train, gas, 60; S = 1_000, initial_params = initial_params)

plot(forecast.scenarios, color = "grey", width = 0.05, label = "")
plot!(y[360:460], label = "ANE", color = "black", xlabel = "Months", ylabel = "GWmed", legend = :topright)
plot!(forecast.quantiles, label = ["Quantiles" "" ""], color = "red", line = :dash)

