function test_score_mean(D::Type{<:Distribution{Univariate,Continuous}}; n::Int = 10^7, seed::Int = 10,
                            atol::Float64 = 1e-3, rtol::Float64 = 1e-3)
    Random.seed!()
    dist = D()
    pars = [params(dist)...]
    avg  = zeros(SDM.num_params(D))
    for i = 1:n
        avg += SDM.score(rand(D(pars...)), D, pars)
    end
    avg ./= n
    @test avg ≈ zeros(SDM.num_params(D)) atol = atol rtol = rtol
end
function test_score_mean(D::Type{<:Distribution}; atol::Float64 = 1e-3, rtol::Float64 = 1e-3)
    @warn("Score of $D is not currently tested because it has countably infinite support.")
    return 
end

function test_loglik(D::Type{<:Distribution}; atol::Float64 = 1e-3, rtol::Float64 = 1e-3,
                     seed::Int = 13, n::Int = 100)
    Random.seed!(seed)
    dist = D()
    y = rand(dist, n)
    pars = [vcat(params(dist)...) for _ in 1:n]
    log_lik = SDM.log_likelihood(D, y, pars, n)
    @test log_lik ≈ -loglikelihood(dist, y) atol = atol rtol = rtol
    return
end

@testset "Score" begin
    for dist in SDM.DISTS
        @testset "$dist" begin
            test_score_mean(dist)
        end
    end
end

@testset "Log Likelihood" begin
    for dist in SDM.DISTS
        @testset "$dist" begin
            test_loglik(dist)
        end
    end
end