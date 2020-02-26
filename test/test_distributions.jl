@testset "Score" begin
    for dist in ScoreDrivenModels.DISTS
        @testset "$dist" begin
            test_score_mean(dist)
        end
    end
end

@testset "Fisher Information" begin
    for dist in ScoreDrivenModels.DISTS
        @testset "$dist" begin
            test_fisher_information(dist)
        end
    end
end

@testset "Log-likelihood" begin
    for dist in ScoreDrivenModels.DISTS
        @testset "$dist" begin
            test_loglik(dist)
        end
    end
end

@testset "Number of parameters" begin
    @test ScoreDrivenModels.num_params(Beta) == 2
    @test ScoreDrivenModels.num_params(Chi) == 1
    @test ScoreDrivenModels.num_params(Chisq) == 1
    @test ScoreDrivenModels.num_params(Exponential) == 1
    @test ScoreDrivenModels.num_params(Gamma) == 2
    @test ScoreDrivenModels.num_params(LogitNormal) == 1
    @test ScoreDrivenModels.num_params(LogNormal) == 1
    @test ScoreDrivenModels.num_params(Poisson) == 1
    @test ScoreDrivenModels.num_params(Weibull) == 2
end
