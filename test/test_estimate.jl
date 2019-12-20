@testset "Estimate" begin
    @testset "Beta" begin
        ω = [0.1, 0.1]
        A = [0.5 0; 0 0.5]
        B = [0.5 0; 0 0.5]
        simulation = simulate_GAS_1_1(Beta, 0.0, ω, A, B, 1)
        @testset "Estimation by passing number of seeds" begin
            # LBFGS
            gas = GAS(1, 1, Beta, 0.0)
            fit!(gas, simulation; verbose = 1, opt_method = SDM.LBFGS(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
            # NelderMead
            gas = GAS(1, 1, Beta, 0.0)
            fit!(gas, simulation; verbose = 1, opt_method = SDM.NelderMead(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
            # IPNewton
            gas = GAS(1, 1, Beta, 0.0)
            fit!(gas, simulation; verbose = 1, opt_method = SDM.IPNewton(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
        end
        @testset "Estimation by passing seeds" begin
            # LBFGS
            gas = GAS(1, 1, Beta, 0.0)
            seeds = [[0.1, 0.1, 0.5, 0.5, 0.5, 0.5]]
            fit!(gas, simulation; verbose = 1, opt_method = SDM.LBFGS(gas, seeds))
            test_coefficients_GAS_1_1(gas, ω, A, B)
            # NelderMead
            gas = GAS(1, 1, Beta, 0.0)
            seeds = [[0.1, 0.1, 0.5, 0.5, 0.5, 0.5]]
            fit!(gas, simulation; verbose = 1, opt_method = SDM.NelderMead(gas, seeds))
            test_coefficients_GAS_1_1(gas, ω, A, B)
            # IPNewton
            gas = GAS(1, 1, Beta, 0.0)
            seeds = [[0.1, 0.1, 0.5, 0.5, 0.5, 0.5]]
            fit!(gas, simulation; verbose = 1, opt_method = SDM.IPNewton(gas, seeds))
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
            fit!(gas, simulation; verbose = 1, opt_method = SDM.LBFGS(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
        end
        @testset "Scaling = 0.5" begin
            ω = [0.1, 0.1]
            A = [0.5 0; 0 0.5]
            B = [0.5 0; 0 0.5]
            simulation = simulate_GAS_1_1(LogNormal, 0.5, ω, A, B, 13)
            gas = GAS(1, 1, LogNormal, 0.5)
            fit!(gas, simulation; verbose = 1, opt_method = SDM.LBFGS(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
        end
        @testset "Scaling = 1.0" begin
            ω = [0.5, 0.5]
            A = [0.1 0; 0 0.1]
            B = [0.1 0; 0 0.1]
            simulation = simulate_GAS_1_1(LogNormal, 1.0, ω, A, B, 3)
            gas = GAS(1, 1, LogNormal, 1.0)
            fit!(gas, simulation; verbose = 1, opt_method = SDM.LBFGS(gas, 3))
            test_coefficients_GAS_1_1(gas, ω, A, B)
        end
        @testset "GAS([1, 12], [1, 12])" begin
            simulation = simulate_GAS_1_12(LogNormal, 0.0, 123)
            gas = GAS([1, 12], [1, 12], LogNormal, 0.0)
            fit!(gas, simulation; verbose = 1, opt_method = SDM.LBFGS(gas, 3))
            test_coefficients_GAS_1_12(gas)
        end
    end
    @testset "GARCH(1, 1)" begin
        # Compare results with ARCHModels
        y = readdlm("$(@__DIR__)/data/BG96.csv")[:]
        test_GARCH_1_1(y, 1, IPNewton)
    end
end
