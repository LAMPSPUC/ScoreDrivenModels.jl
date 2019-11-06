function test_dynamic(sigma::Float64, lags::Int; seed::Int = 12, atol::Float64 = 1e-1, rtol::Float64 = 1e-3, n::Int = 100)
    Random.seed!(seed)
    dist = Normal(0, sigma^2)
    obs_dynamic = kron(ones(n), collect(1:lags)) + rand(dist, n*lags)
    gas_lag_lag = GAS(lags, lags, Normal, 0.0)
    initial_params = dynamic_initial_params(obs_dynamic, gas_lag_lag)
    for i in eachindex(initial_params)
        @test initial_params[i][1] ≈ i atol = atol rtol = rtol
        @test initial_params[i][2] ≈ sigma atol = atol rtol = rtol
    end
end

@testset "Inital params" begin
    @testset "stationary initial params" begin
        gas_1_1 = GAS(1, 1, Normal, 0.0)
        gas_1_1.ω = [1; 1]
        gas_1_1.B[1] = [0.8 0;0 0.5]
        initial_params = stationary_initial_params(gas_1_1)
        @test initial_params[1][1] ≈ 5 atol = 1e-2 rtol = 1e-2
        @test initial_params[1][2] ≈ 2 atol = 1e-2 rtol = 1e-2

        gas_1_2 = GAS(1, 2, Normal, 0.0)
        gas_1_2.ω = [1; 1]
        gas_1_2.B[1] = [0.8 0;0 0.5]
        initial_params = stationary_initial_params(gas_1_2)
        @test initial_params[1][1] ≈ 5 atol = 1e-2 rtol = 1e-2
        @test initial_params[1][2] ≈ 2 atol = 1e-2 rtol = 1e-2
        @test initial_params[2][1] ≈ 5 atol = 1e-2 rtol = 1e-2
        @test initial_params[2][2] ≈ 2 atol = 1e-2 rtol = 1e-2
    end
    @testset "Dynamic initial params" begin
        test_dynamic(0.2, 12)
        test_dynamic(0.1, 3)
    end
end