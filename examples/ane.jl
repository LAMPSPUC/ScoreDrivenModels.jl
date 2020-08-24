using ScoreDrivenModels, Plots, DelimitedFiles, Dates, Random

dates = collect(Date(1961):Month(1):Date(2000, 12))
Random.seed!(123);
y = vec(readdlm("../test/data/ane_northeastern.csv"));
y_train = y[1:400];
gas = Model([1, 2, 11, 12], [1, 2, 11, 12], LogNormal, 0.0; time_varying_params=[1]);
initial_params = dynamic_initial_params(y_train, gas);
f = ScoreDrivenModels.fit!(gas, y_train; initial_params=initial_params);
estimation_stats = fit_stats(f)

forec = ScoreDrivenModels.forecast(y_train, gas, 60; S=1_000, initial_params=initial_params)

y_test = y[401:460]
p2 = plot(dates[401:460], forec.observation_scenarios, color="grey", width=0.05, label="", ylims=(0, 70))
plot!(p2, dates[360:460], y[360:460], label="ANE", color="black", xlabel="Months", ylabel="GWmed", legend=:topright)
plot!(p2, dates[401:460], forec.observation_quantiles, label=["Quantiles" "" ""], color="red", line=:dash)
