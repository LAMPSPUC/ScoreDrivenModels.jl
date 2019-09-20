function simulate_GAS_Sarima_1_1(dist::Distribution, scaling::Float64)
    # Create model
    Random.seed!(123)
    vec = 0.1*ones(length(params(dist)))
    ω = vec
    A = convert(Matrix{Float64}, Diagonal(vec))
    B = convert(Matrix{Float64}, Diagonal(vec))  

    sd_model = GAS_Sarima(ω, A, B, dist, scaling)

    # Simulate 1000 observations
    serie_simulated, param_simulated = simulate(sd_model, 1000)

    return serie_simulated
end

function test_coefficients_GAS_Sarima_1_1(vec::Vector, i::Int; atol = 1e-2, rtol = 1e-1)
    @test vec[i] ≈ 0.1 atol = atol rtol = rtol
end
function test_coefficients_GAS_Sarima_1_1(mat::Matrix, i::Int; atol = 1e-2, rtol = 1e-1)
    @test mat[i, i] ≈ 0.1 atol = atol rtol = rtol
end

function test_estimation_GAS_Sarima_1_1(sd_model, simulation::Vector{T}) where T
    estimate_GAS_Sarima!(sd_model, simulation; verbose = 2, 
                         random_seeds_lbfgs = ScoreDrivenModels.RandomSeedsLBFGS(10, ScoreDrivenModels.dimension_unkowns(sd_model)))

    for i in eachindex(sd_model.ω)
        test_coefficients_GAS_Sarima_1_1(sd_model.ω, i)
        test_coefficients_GAS_Sarima_1_1(sd_model.A, i)
        test_coefficients_GAS_Sarima_1_1(sd_model.B, i)
    end
    return 
end


@testset "Estimate" begin
    @testset "Normal" begin
        simulation = simulate_GAS_Sarima(Normal(), 0.0)
        sd_model = GAS_Sarima(1, 1, Normal(), 0.0)
        test_estimation_GAS_Sarima_1_1(sd_model, simulation)
    end
    @testset "Poisson" begin
        simulation = simulate_GAS_Sarima(Poisson(), 0.0)
        sd_model = GAS_Sarima(1, 1, Poisson(), 0.0)
        test_estimation_GAS_Sarima_1_1(sd_model, simulation)
    end
    @testset "Beta" begin
        simulation = simulate_GAS_Sarima(Beta(), 0.0)
        sd_model = GAS_Sarima(1, 1, Beta(), 0.0)
        test_estimation_GAS_Sarima_1_1(sd_model, simulation)
    end
end


