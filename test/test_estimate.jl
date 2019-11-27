function simulate_GAS_1_1(D::Type{<:Distribution}, scaling::Float64, ω::Vector{T}, A::Matrix{T}, 
                            B::Matrix{T}, seed::Int) where T
    Random.seed!(seed)
    gas = GAS(1, 1, D, scaling)

    gas.ω = ω
    gas.A[1] = A
    gas.B[1] = B
    series, param = simulate(gas, 1000)

    return series
end

function simulate_GAS_1_12(D::Type{<:Distribution}, scaling::Float64)
    Random.seed!(123)
    v = [0.1, 0.1]
    gas = GAS([1, 12], [1, 12], D, scaling)

    gas.ω = v
    gas.A[1]  = convert(Matrix{Float64}, Diagonal(3*v))
    gas.A[12] = convert(Matrix{Float64}, Diagonal(-3*v))
    gas.B[1]  = convert(Matrix{Float64}, Diagonal(3*v))
    gas.B[12] = convert(Matrix{Float64}, Diagonal(-3*v))

    series, param = simulate(gas, 5000)

    return series
end

function test_coefficients_GAS_1_1(gas::GAS{D, T}, ω::Vector{T}, A::Matrix{T}, B::Matrix{T}; 
                                    atol = 1e-1, rtol = 1e-1) where {D <: Distribution, T}
    @test gas.ω[1] ≈ ω[1] atol = atol rtol = rtol
    @test gas.ω[2] ≈ ω[2] atol = atol rtol = rtol
    @test gas.A[1][1, 1] ≈ A[1, 1] atol = atol rtol = rtol
    @test gas.A[1][2, 2] ≈ A[2, 2] atol = atol rtol = rtol
    @test gas.B[1][1, 1] ≈ B[1, 1] atol = atol rtol = rtol
    @test gas.B[1][2, 2] ≈ B[2, 2] atol = atol rtol = rtol
    return
end

function test_coefficients_GAS_1_12(gas::GAS{D, T}; atol = 1e-1, rtol = 1e-1) where {D <: Distribution, T}
    @test gas.ω[1] ≈ 0.1 atol = atol rtol = rtol
    @test gas.ω[2] ≈ 0.1 atol = atol rtol = rtol
    @test gas.A[1][1, 1] ≈ 0.3 atol = atol rtol = rtol
    @test gas.A[1][2, 2] ≈ 0.3 atol = atol rtol = rtol
    @test gas.A[12][1, 1] ≈ -0.3 atol = atol rtol = rtol
    @test gas.A[12][2, 2] ≈ -0.3 atol = atol rtol = rtol
    @test gas.B[1][1, 1] ≈ 0.3 atol = atol rtol = rtol
    @test gas.B[1][2, 2] ≈ 0.3 atol = atol rtol = rtol
    @test gas.B[12][1, 1] ≈ -0.3 atol = atol rtol = rtol
    @test gas.B[12][2, 2] ≈ -0.3 atol = atol rtol = rtol
    return
end

@testset "Estimate" begin
    @testset "Beta" begin
        ω = [0.1, 0.1]
        A = [0.5 0; 0 0.5]
        B = [0.5 0; 0 0.5]
        simulation = simulate_GAS_1_1(Beta, 0.0, ω, A, B, 13)
        @testset "Estimation by passing number of seeds" begin
            gas = GAS(1, 1, Beta, 0.0)
            estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
        end
        @testset "Estimation by passing seeds" begin
            gas = GAS(1, 1, Beta, 0.0)
            seeds = [[0.1, 0.1, 0.5, 0.5, 0.5, 0.5]]
            estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, seeds))
            test_coefficients_GAS_1_1(gas, ω, A, B)
        end
    end

    @testset "Lognormal" begin
        @testset "Scaling = 0.0" begin
            ω = [0.1, 0.1]
            A = [0.5 0; 0 0.5]
            B = [0.5 0; 0 0.5]
            simulation = simulate_GAS_1_1(LogNormal, 0.0, ω, A, B, 13)
            gas = GAS(1, 1, LogNormal, 0.0)
            estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
        end
        # @testset "Scaling = 0.5" begin
        #     gas = GAS(1, 1, LogNormal, 0.5)
        #     estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
        #     test_coefficients_GAS_1_1(gas)
        # end
        # @testset "Scaling = 1.0" begin
        #     ω = [0.5, 0.5]
        #     A = [0.1 0; 0 0.1]
        #     B = [0.1 0; 0 0.1]
        #     simulation = simulate_GAS_1_1(LogNormal, 1.0, ω, A, B, 123)
        #     gas = GAS(1, 1, LogNormal, 1.0)
        #     estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
        #     test_coefficients_GAS_1_1(gas)
        # end
        @testset "GAS([1, 12], [1, 12])" begin
            simulation = simulate_GAS_1_12(LogNormal, 0.0)
            gas = GAS([1, 12], [1, 12], LogNormal, 0.0)
            estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
            test_coefficients_GAS_1_12(gas)
        end
    end
end