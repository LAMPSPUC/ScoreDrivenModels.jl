using ScoreDrivenModels

ScoreDrivenModel(1, 2, LogNormal, 0.5)

ScoreDrivenModel(1, 2, LogNormal, 0.5; time_varying_params = [1])

ScoreDrivenModel([1, 12], [1, 12], LogNormal, 0.5)
