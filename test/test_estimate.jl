function simulate_GAS_Sarima_1_1(dist::Distribution, scaling::Float64)
    # Create model
    Random.seed!(123)
    vec = 0.5*ones(length(params(dist)))

    gas_sarima = GAS_Sarima(1, 1, dist, scaling)

    gas_sarima.ω = vec
    gas_sarima.A[1] = convert(Matrix{Float64}, Diagonal(vec))
    gas_sarima.B[1] = convert(Matrix{Float64}, Diagonal(vec))  

    # Simulate 1000 observations
    serie_simulated, param_simulated = simulate(gas_sarima, 1000)

    return serie_simulated
end

function test_coefficients_GAS_Sarima_1_1(vec::Vector, i::Int; atol = 1e-2, rtol = 1e-1)
    @test vec[i] ≈ 0.1 atol = atol rtol = rtol
end
function test_coefficients_GAS_Sarima_1_1(mat::Matrix, i::Int; atol = 1e-2, rtol = 1e-1)
    @test mat[i, i] ≈ 0.1 atol = atol rtol = rtol
end

function test_estimation_GAS_Sarima_1_1(gas_sarima, simulation::Vector{T}) where T
    gas_sarima = GAS_Sarima(1, 1, Beta(), 0.0)
    estimate_GAS_Sarima!(gas_sarima, simulation; verbose = 0,
                         random_seeds_lbfgs = ScoreDrivenModels.RandomSeedsLBFGS(5, ScoreDrivenModels.dim_unknowns(gas_sarima)))

    for i in eachindex(gas_sarima.ω)
        test_coefficients_GAS_Sarima_1_1(gas_sarima.ω, i)
        test_coefficients_GAS_Sarima_1_1(gas_sarima.A, i)
        test_coefficients_GAS_Sarima_1_1(gas_sarima.B, i)
    end
    return 
end


@testset "Estimate" begin
    @testset "Normal" begin
        simulation = simulate_GAS_Sarima_1_1(Normal(), 0.0)
        gas_sarima = GAS_Sarima(1, 1, Normal(), 0.0)
        test_estimation_GAS_Sarima_1_1(gas_sarima, simulation)
    end
    @testset "Poisson" begin
        simulation = simulate_GAS_Sarima_1_1(Poisson(), 0.0)
        gas_sarima = GAS_Sarima(1, 1, Poisson(), 0.0)
        test_estimation_GAS_Sarima_1_1(gas_sarima, simulation)
    end
    @testset "Beta" begin
        simulation = simulate_GAS_Sarima_1_1(Beta(), 0.0)
        gas_sarima = GAS_Sarima(1, 1, Beta(), 0.0)
        test_estimation_GAS_Sarima_1_1(gas_sarima, simulation)
    end
end