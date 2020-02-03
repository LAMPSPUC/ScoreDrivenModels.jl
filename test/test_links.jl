@testset "Links" begin
    Random.seed!(10)
    atol = 1e-7 
    rtol = 1e-7

    for x in rand(10)
        link = ScoreDrivenModels.IdentityLink
        @test ScoreDrivenModels.link(link, ScoreDrivenModels.unlink(link, x)) ≈ x atol = atol rtol = rtol

        link = ScoreDrivenModels.LogLink
        lb = rand()
        @test ScoreDrivenModels.link(link, ScoreDrivenModels.unlink(link, x, lb), lb) ≈ x atol = atol rtol = rtol

        link = ScoreDrivenModels.LogitLink
        lb = rand()
        ub = rand() + lb
        @test ScoreDrivenModels.link(link, ScoreDrivenModels.unlink(link, x, lb, ub), lb, ub) ≈ x atol = atol rtol = rtol
    end
end