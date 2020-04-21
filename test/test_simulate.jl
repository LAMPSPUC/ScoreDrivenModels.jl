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

        scenarios = simulate(y, gas, 50, 1000)
        @test maximum(scenarios) < 1e5
        @test minimum(scenarios) > 0

        forecast = forecast_quantiles(y, gas, 50)
        @test all([forecast.quantiles[i, 1] .< forecast.quantiles[i, 2] .< forecast.quantiles[i, 3] for i in 1:50])
        @test maximum(forecast.quantiles) < 1e3
        @test minimum(forecast.quantiles) > 1e-4
    end
end
