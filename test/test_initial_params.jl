@testset "Inital params" begin
    @testset "stationary initial params" begin
        gas_1_1 = GAS(1, 1, Normal, 0.0)
        gas_1_1.ω = [1; 1]
        gas_1_1.B[1] = [0.8 0;0 0.5]
        initial_params = stationary_initial_params_tilde(gas_1_1)
        @test initial_params[1][1] ≈ 5 atol = 1e-2 rtol = 1e-2
        @test initial_params[1][2] ≈ 2 atol = 1e-2 rtol = 1e-2

        gas_1_2 = GAS(1, 2, Normal, 0.0)
        gas_1_2.ω = [1; 1]
        gas_1_2.B[1] = [0.8 0;0 0.5]
        initial_params = stationary_initial_params_tilde(gas_1_2)
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