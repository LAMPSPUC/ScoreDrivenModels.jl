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

@testset "Link interface" begin
    for dist in ScoreDrivenModels.DISTS
        @testset "$dist" begin
            test_link_interfaces(dist)
        end
    end
end

@testset "Utils" begin
    for dist in ScoreDrivenModels.DISTS
        @testset "$dist" begin
            test_dist_utils(dist)
        end
    end
end

@testset "Common interface" begin
    test_distribution_common_interface()
end