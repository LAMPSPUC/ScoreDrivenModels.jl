@testset "Links" begin
    Random.seed!(10)
    x = rand()
    atol = 1e-7 
    rtol = 1e-7

    link = SDM.IdentityLink
    @test SDM.param_to_param_tilde(link, SDM.param_tilde_to_param(link, x)) ≈ x atol = atol rtol = rtol

    link = SDM.ExponentialLink
    lb = 1.0
    @test SDM.param_to_param_tilde(link, SDM.param_tilde_to_param(link, x, lb), lb) ≈ x atol = atol rtol = rtol
    lb = 0.0
    @test SDM.param_to_param_tilde(link, SDM.param_tilde_to_param(link, x, lb), lb) ≈ x atol = atol rtol = rtol
end