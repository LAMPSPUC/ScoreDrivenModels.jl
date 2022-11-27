@testset "CrossValidation" begin
    ω = [0.1, 0.1]
    A = [0.5 0; 0 0.5]
    B = [0.5 0; 0 0.5]
    simulation = simulate_GAS_1_1(Normal, 0.0, ω, A, B, 1)
    gas = ScoreDrivenModel(1, 1, Normal, 0.0)
    bac = cross_validation(gas, simulation, 10, 4985)
end
