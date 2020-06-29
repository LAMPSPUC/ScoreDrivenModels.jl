@testset "Simulation" begin
    @testset "LogNormal" begin
        Random.seed!(123)
        y = simulate_GAS_1_12(LogNormal, 0.0, 123)

        v = [0.1, 0.1]
        gas = Model([1, 12], [1, 12], LogNormal, 0.0)
        gas.ω = v
        gas.A[1]  = convert(Matrix{Float64}, Diagonal(3*v))
        gas.A[12] = convert(Matrix{Float64}, Diagonal(-3*v))
        gas.B[1]  = convert(Matrix{Float64}, Diagonal(3*v))
        gas.B[12] = convert(Matrix{Float64}, Diagonal(-3*v))

        scenarios_observations, scenarios_params = simulate(y, gas, 50, 1000)
        @test maximum(scenarios_observations) < 1e5
        @test minimum(scenarios_observations) > 0

        forec = forecast(y, gas, 50)
        @test all([forec.observation_quantiles[i, 1] .< 
                   forec.observation_quantiles[i, 2] .< 
                   forec.observation_quantiles[i, 3] for i in 1:50])
        @test maximum(forec.observation_quantiles) < 1e3
        @test minimum(forec.observation_quantiles) > 1e-4
    end
end
