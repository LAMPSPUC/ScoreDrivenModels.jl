@testset "Visualizations" begin
    ω = [0.1, 0.1]
    A = [0.5 0; 0 0.5]
    B = [0.5 0; 0 0.5]
    simulation = simulate_GAS_1_1(Normal, 0.0, ω, A, B, 1)
    # LBFGS
    gas = Model(1, 1, Normal, 0.0)
    f = fit!(gas, simulation; verbose = 2, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), f)
    @test length(rec) == 7
    ω = [0.1, 0.1]
    A = [0.5 0; 0 0.5]
    B = [0.5 0; 0 0.5]
    simulation = simulate_GAS_1_1(Normal, 0.0, ω, A, B, 1)
    gas = ScoreDrivenModels.Model(1, 1, Normal, 0.0)
    bac = cross_validation(gas, simulation, 10, 4985)
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), bac, "name")
    @test length(rec) == 2

    simulation = simulate_GAS_1_1(Normal, 0.0, ω, A, B, 1)
    gas = ScoreDrivenModels.Model(1, 1, Normal, 0.0)
    fit!(gas, simulation)
    forec = forecast(simulation, gas, 50)
    rec = RecipesBase.apply_recipe(Dict{Symbol, Any}(), simulation, forec)
    @test length(rec) == 4
end