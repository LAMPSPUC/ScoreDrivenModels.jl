using ScoreDrivenModels, DelimitedFiles, Random, Statistics
Random.seed!(123); 
y = vec(readdlm("./test/data/cpichg.csv"));

gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params=[1; 2]);
f = ScoreDrivenModels.fit!(gas, y);

gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params=[1; 2]);
f = fit!(gas, y; verbose=2);

gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params=[1; 2]);
f = fit!(gas, y; opt_method=LBFGS(gas, 5));

estimation_stats = fit_stats(f)

forec = forecast(y, gas, 12);
forec.parameter_forecast
forec.observation_scenarios

param = score_driven_recursion(gas, y)

model = local_level(y)
ss = statespace(model)
ss.filter.a

plot([y])
plot([y, param[:, 1], ss.filter.a[:]])

cpi = readdlm("/Users/guilhermebodin/Downloads/CPIAUCSL.csv", ',')
using ShiftedArrays
dates = Date.(cpi[2:end, 1])
cpi_vals = float.(cpi[2:end, 2])
d_log_cpi_vals = diff(log.(cpi_vals)) * 100
plot(dates[2:end], d_log_cpi_vals)

data = d_log_cpi_vals[end - 300:end]
dates_data = dates[end - 300:end]

gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params=[1; 2]);
f = ScoreDrivenModels.fit!(gas, data; verbose=2);
param = score_driven_recursion(gas, data)

model = local_level(data)
ss = statespace(model)
ss.filter.a

plot([data, param[:, 1], ss.filter.a])
