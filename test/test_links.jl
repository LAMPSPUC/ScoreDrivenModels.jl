@testset "Links" begin
    Random.seed!(10)
    atol = 1e-7 
    rtol = 1e-7

    for x in rand(10)
        link = GAS.IdentityLink
        @test GAS.link(link, GAS.unlink(link, x)) ≈ x atol = atol rtol = rtol

        link = GAS.LogLink
        lb = rand()
        @test GAS.link(link, GAS.unlink(link, x, lb), lb) ≈ x atol = atol rtol = rtol

        link = GAS.LogitLink
        lb = rand()
        ub = rand() + lb
        @test GAS.link(link, GAS.unlink(link, x, lb, ub), lb, ub) ≈ x atol = atol rtol = rtol
    end
end