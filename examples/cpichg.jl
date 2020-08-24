using ScoreDrivenModels, DelimitedFiles, Random, Statistics
Random.seed!(123); 
y = vec(readdlm("./test/data/cpichg.csv"));

gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params=[1, 2]);
f = ScoreDrivenModels.fit!(gas, y);

gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params=[1, 2]);
f = fit!(gas, y; verbose=2);

gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params=[1; 2]);
f = fit!(gas, y; opt_method=LBFGS(gas, 5));

estimation_stats = fit_stats(f)

forec = forecast(y, gas, 12);
forec.parameter_forecast
forec.observation_scenarios

param = score_driven_recursion(gas, y)