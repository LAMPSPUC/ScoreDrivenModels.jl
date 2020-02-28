using ScoreDrivenModels, DelimitedFiles

y = readdlm("./test/data/cpichg.csv")[:]

# Define a model where only \mu and \sigma^2 is varying over time
gas = Model(1, 1, TDistLocationScale, 0.0, time_varying_params = [1; 2])

# Estimate the model
@time f = fit!(gas, y)

estimation_stats = fit_stats(f)