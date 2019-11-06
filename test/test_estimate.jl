function simulate_GAS_Sarima_1_1(D::Type{Beta}, scaling::Float64)
    # Create model
    Random.seed!(13)
    vec = [0.1 ; 0.1]

    gas = GAS(1, 1, D, scaling)

    gas.ω = vec
    gas.A[1] = convert(Matrix{Float64}, Diagonal(5*vec))
    gas.B[1] = convert(Matrix{Float64}, Diagonal(5*vec))  

    # Simulate 1000 observations
    serie_simulated, param_simulated = simulate(gas, 1000)

    return serie_simulated
end

function test_coefficients_GAS_Sarima_1_1(gas::GAS{Beta, T}; atol = 1e-1, rtol = 1e-1) where T
    @test gas.ω[1] ≈ 0.1 atol = atol rtol = rtol
    @test gas.ω[2] ≈ 0.1 atol = atol rtol = rtol
    @test gas.A[1][1, 1] ≈ 0.5 atol = atol rtol = rtol
    @test gas.A[1][2, 2] ≈ 0.5 atol = atol rtol = rtol
    @test gas.B[1][1, 1] ≈ 0.5 atol = atol rtol = rtol
    @test gas.B[1][2, 2] ≈ 0.5 atol = atol rtol = rtol
    return
end

function test_estimation_GAS_Sarima_1_1(gas::GAS{D, T}, simulation::Vector{T}) where {D, T}
    estimate!(gas, simulation; verbose = 1,
                         opt_method = ScoreDrivenModels.LBFGS(gas, 5))

    test_coefficients_GAS_Sarima_1_1(gas)
    return 
end

@testset "Estimate" begin
    @testset "Beta" begin
        simulation = simulate_GAS_Sarima_1_1(Beta, 0.0)
        gas = GAS(1, 1, Beta, 0.0)
        test_estimation_GAS_Sarima_1_1(gas, simulation)
    end
end