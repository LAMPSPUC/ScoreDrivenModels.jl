function simulate_GAS_Sarima_1_1(dist::Beta, scaling::Float64)
    # Create model
    Random.seed!(13)
    vec = [0.1 ; 0.1]

    gas_sarima = GAS_Sarima(1, 1, dist, scaling)

    gas_sarima.ω = vec
    gas_sarima.A[1] = convert(Matrix{Float64}, Diagonal(5*vec))
    gas_sarima.B[1] = convert(Matrix{Float64}, Diagonal(5*vec))  

    # Simulate 1000 observations
    serie_simulated, param_simulated = simulate(gas_sarima, 1000)

    return serie_simulated
end

function test_coefficients_GAS_Sarima_1_1(dist::Beta, gas_sarima::GAS_Sarima; atol = 1e-1, rtol = 1e-1)
    @test gas_sarima.ω[1] ≈ 0.1 atol = atol rtol = rtol
    @test gas_sarima.ω[2] ≈ 0.1 atol = atol rtol = rtol
    @test gas_sarima.A[1][1, 1] ≈ 0.5 atol = atol rtol = rtol
    @test gas_sarima.A[1][2, 2] ≈ 0.5 atol = atol rtol = rtol
    @test gas_sarima.B[1][1, 1] ≈ 0.5 atol = atol rtol = rtol
    @test gas_sarima.B[1][2, 2] ≈ 0.5 atol = atol rtol = rtol
    return
end

function test_estimation_GAS_Sarima_1_1(gas_sarima::GAS_Sarima, simulation::Vector{T}) where T
    estimate_GAS_Sarima!(gas_sarima, simulation; verbose = 0,
                         random_seeds_lbfgs = ScoreDrivenModels.RandomSeedsLBFGS(5, ScoreDrivenModels.dim_unknowns(gas_sarima)))

    test_coefficients_GAS_Sarima_1_1(gas_sarima.dist, gas_sarima)
    return 
end

@testset "Estimate" begin
    @testset "Beta" begin
        simulation = simulate_GAS_Sarima_1_1(Beta(), 0.0)
        gas_sarima = GAS_Sarima(1, 1, Beta(), 0.0)
        test_estimation_GAS_Sarima_1_1(gas_sarima, simulation)
    end
end