@testset "Links" begin
    Random.seed!(10)
    atol = 1e-7 
    rtol = 1e-7

    for x in rand(10)
        link = SDM.IdentityLink
        @test SDM.link(link, SDM.unlink(link, x)) ≈ x atol = atol rtol = rtol

        link = SDM.LogLink
        lb = rand()
        @test SDM.link(link, SDM.unlink(link, x, lb), lb) ≈ x atol = atol rtol = rtol

        link = SDM.LogitLink
        lb = rand()
        ub = rand() + lb
        @test SDM.link(link, SDM.unlink(link, x, lb, ub), lb, ub) ≈ x atol = atol rtol = rtol
    end
end