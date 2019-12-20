@testset "Links" begin
    Random.seed!(10)
    atol = 1e-7 
    rtol = 1e-7

    for x in rand(10)
        link = IdentityLink()
        @test SDM.link(link, SDM.unlink(link, x)) ≈ x atol = atol rtol = rtol

        lb = rand()
        link = LogLink(lb)
        @test SDM.link(link, SDM.unlink(link, x)) ≈ x atol = atol rtol = rtol

        lb = rand()
        ub = rand() + lb
        link = LogitLink(lb, ub)
        @test SDM.link(link, SDM.unlink(link, x)) ≈ x atol = atol rtol = rtol
    end
end