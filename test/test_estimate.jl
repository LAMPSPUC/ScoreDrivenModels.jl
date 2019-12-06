@testset "Estimate" begin
    @testset "Beta" begin
        ω = [0.1, 0.1]
        A = [0.5 0; 0 0.5]
        B = [0.5 0; 0 0.5]
        simulation = simulate_GAS_1_1(Beta, 0.0, ω, A, B, 1)
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

    # @testset "Lognormal" begin
    #     @testset "Scaling = 0.0" begin
    #         ω = [0.1, 0.1]
    #         A = [0.5 0; 0 0.5]
    #         B = [0.5 0; 0 0.5]
    #         simulation = simulate_GAS_1_1(LogNormal, 0.0, ω, A, B, 13)
    #         gas = GAS(1, 1, LogNormal, 0.0)
    #         estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
    #         test_coefficients_GAS_1_1(gas, ω, A, B)
    #     end
    #     @testset "Scaling = 0.5" begin
    #         ω = [0.1, 0.1]
    #         A = [0.5 0; 0 0.5]
    #         B = [0.5 0; 0 0.5]
    #         simulation = simulate_GAS_1_1(LogNormal, 0.5, ω, A, B, 13)
    #         gas = GAS(1, 1, LogNormal, 0.5)
    #         estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
    #         test_coefficients_GAS_1_1(gas, ω, A, B)
    #     end
    #     @testset "Scaling = 1.0" begin
    #         ω = [0.5, 0.5]
    #         A = [0.1 0; 0 0.1]
    #         B = [0.1 0; 0 0.1]
    #         simulation = simulate_GAS_1_1(LogNormal, 1.0, ω, A, B, 3)
    #         gas = GAS(1, 1, LogNormal, 1.0)
    #         estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
    #         test_coefficients_GAS_1_1(gas, ω, A, B)
    #     end
    #     @testset "GAS([1, 12], [1, 12])" begin
    #         simulation = simulate_GAS_1_12(LogNormal, 0.0, 123)
    #         gas = GAS([1, 12], [1, 12], LogNormal, 0.0)
    #         estimate!(gas, simulation; verbose = 1, opt_method = ScoreDrivenModels.LBFGS(gas, 3))
    #         test_coefficients_GAS_1_12(gas)
    #     end
    # end
end