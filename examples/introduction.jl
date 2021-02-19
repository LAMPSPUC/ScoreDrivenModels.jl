using ScoreDrivenModels

Model(1, 2, LogNormal, 0.5)

Model(1, 2, LogNormal, 0.5; time_varying_params = [1])

Model([1, 12], [1, 12], LogNormal, 0.5)