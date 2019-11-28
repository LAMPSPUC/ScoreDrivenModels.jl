@testset "Links" begin
    Random.seed!(10)
    x = rand()
    atol = 1e-7 
    rtol = 1e-7

    link = SDM.IdentityLink
    @test SDM.link(link, SDM.unlink(link, x)) ≈ x atol = atol rtol = rtol

    link = SDM.LogLink
    lb = 1.0
    @test SDM.link(link, SDM.unlink(link, x, lb), lb) ≈ x atol = atol rtol = rtol
    lb = 0.0
    @test SDM.link(link, SDM.unlink(link, x, lb), lb) ≈ x atol = atol rtol = rtol
end