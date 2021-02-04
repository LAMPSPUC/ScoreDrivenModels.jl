const FI_NOT_IMPLEMENTED = [Weibull; BetaFourParameters; NegativeBinomial]

struct FakeDist{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    foo::T
end

function instantiate_dist(D::Type{<:Distribution})
    if D == ScoreDrivenModels.TDistLocationScale
        params = [1.0 2.0 10]
        return ScoreDrivenModels.update_dist(D, params, 1)
    elseif D == ScoreDrivenModels.TDist
        params = [10][:, :]
        return ScoreDrivenModels.update_dist(D, params, 1)
    elseif D == ScoreDrivenModels.BetaFourParameters
        params = [0.0 10.0 2.0 3]
        return ScoreDrivenModels.update_dist(D, params, 1)
    else
        params = 0.5 * ones(1, ScoreDrivenModels.num_params(D))
        return ScoreDrivenModels.update_dist(D, params, 1)
    end
end

function test_score_mean(D::Type{<:Distribution}; n::Int = 10^7, seed::Int = 10,
                            atol::Float64 = 1e-3, rtol::Float64 = 1e-3)
    Random.seed!(seed)
    dist = instantiate_dist(D)
    pars = permutedims([ScoreDrivenModels.params_sdm(dist)...])
    n_params = ScoreDrivenModels.num_params(D)
    avg  = zeros(1, n_params)
    obs = rand(dist, n)
    score_aux = ones(1, n_params)
    for i = 1:n
        ScoreDrivenModels.score!(score_aux, obs[i], D, pars, 1)
        avg .+= score_aux
    end
    avg ./= n
    @test avg ≈ zeros(1, ScoreDrivenModels.num_params(D)) atol = atol rtol = rtol
end

function test_fisher_information(D::Type{<:Distribution}; n::Int = 10^6, seed::Int = 10,
                                 atol::Float64 = 1e-2, rtol::Float64 = 1e-2)
    Random.seed!(seed)
    dist = instantiate_dist(D)
    pars = permutedims([ScoreDrivenModels.params_sdm(dist)...])
    n_params = ScoreDrivenModels.num_params(D)
    var_terms  = zeros(n_params, n_params)
    obs = rand(dist, n)
    score_aux = ones(1, n_params)
    for i = 1:n
        ScoreDrivenModels.score!(score_aux, obs[i], D, pars, 1)
        var_terms .+= score_aux' * score_aux
    end
    var_terms ./= n
    aux_lin_alg = AuxiliaryLinAlg{Float64}(n_params)

    # Some distributions might not have the fisher information yet available
    if D in FI_NOT_IMPLEMENTED
        return 
    else
        ScoreDrivenModels.fisher_information!(aux_lin_alg, D, pars, 1)
    end

    @test var_terms ≈ aux_lin_alg.fisher atol = atol rtol = rtol
end

function test_loglik(D::Type{<:Distribution}; atol::Float64 = 1e-3, rtol::Float64 = 1e-3,
                     seed::Int = 13, n::Int = 100)
    Random.seed!(seed)
    dist = instantiate_dist(D)
    y = rand(dist, n)
    pars = Matrix{Float64}(undef, n, ScoreDrivenModels.num_params(D))
    pars_dist = ScoreDrivenModels.params_sdm(dist)
    for t in axes(pars, 1), p in axes(pars, 2)
        pars[t, p] = pars_dist[p]
    end
    log_lik = ScoreDrivenModels.log_likelihood(D, y, pars, n)
    @test log_lik ≈ -loglikelihood(dist, y) atol = atol rtol = rtol
    return
end

function test_link_interfaces(D::Type{<:Distribution})
    n_params = ScoreDrivenModels.num_params(D)
    t = 1
    param_tilde = zeros(1, n_params)
    param = ones(1, n_params)
    ScoreDrivenModels.link!(param_tilde, D, param, t)
    ScoreDrivenModels.unlink!(param, D, param_tilde, t)
    @test param == ones(1, n_params)

    # No tests for jacobian_link! just pass through the function
    aux = AuxiliaryLinAlg{Float64}(n_params)
    ScoreDrivenModels.jacobian_link!(aux, D, param, t)
end

function test_dist_utils(D::Type{<:Distribution})
    # num_params
    dist = instantiate_dist(D) 
    n_pars = ScoreDrivenModels.num_params(D)
    @test n_pars == length(ScoreDrivenModels.params_sdm(dist))
     
    # update_dist
    t = 1
    pars = permutedims([ScoreDrivenModels.params_sdm(dist)...])
    updated_dist = ScoreDrivenModels.update_dist(D, pars, t)
    @test typeof(updated_dist) <: D
end

function test_distribution_common_interface()
    score_til = ones(1, 1)
    param = ones(1, 1)
    param_tilde = ones(1, 1)
    aux = AuxiliaryLinAlg{Float64}(1)
    t = 1
    n = 1
    y_f64 = 1.0
    y_i64 = 1
    @test_throws ErrorException ScoreDrivenModels.score!(score_til, y_f64, FakeDist, param, t)
    @test_throws ErrorException ScoreDrivenModels.score!(score_til, y_i64, FakeDist, param, t)
    @test_throws ErrorException ScoreDrivenModels.fisher_information!(aux, FakeDist, param, t)
    @test_throws ErrorException ScoreDrivenModels.log_likelihood(FakeDist, [y_f64], param, n)
    @test_throws ErrorException ScoreDrivenModels.link!(param_tilde, FakeDist, param, t)
    @test_throws ErrorException ScoreDrivenModels.unlink!(param, FakeDist, param_tilde, t)
    @test_throws ErrorException ScoreDrivenModels.jacobian_link!(aux, FakeDist, param, t)
    @test_throws ErrorException ScoreDrivenModels.update_dist(FakeDist, param, t)
    @test_throws ErrorException ScoreDrivenModels.params_sdm(FakeDist(1))
    @test_throws ErrorException ScoreDrivenModels.num_params(FakeDist)
end

function test_dynamic(sigma::Float64, lags::Int; seed::Int = 12, atol::Float64 = 1e-1, rtol::Float64 = 1e-3, n::Int = 100)
    Random.seed!(seed)
    dist = Normal(0, sigma^2)
    obs_dynamic = kron(ones(n), collect(1:lags)) + rand(dist, n*lags)
    gas_lag_lag = Model(lags, lags, Normal, 0.0)
    initial_params = dynamic_initial_params(obs_dynamic, gas_lag_lag)
    for i in axes(initial_params, 1)
        @test initial_params[i, 1] ≈ i atol = atol rtol = rtol
        @test initial_params[i, 2] ≈ sigma^2 atol = atol rtol = rtol
    end
end

function simulate_GAS_1_1(D::Type{<:Distribution}, scaling::Float64, ω::Vector{T}, A::Matrix{T},
                            B::Matrix{T}, seed::Int) where T
    Random.seed!(seed)
    gas = ScoreDrivenModels.Model(1, 1, D, scaling)

    gas.ω = ω
    gas.A[1] = A
    gas.B[1] = B
    series, param = simulate_recursion(gas, 5000)

    return series
end

function simulate_GAS_1_12(D::Type{<:Distribution}, scaling::Float64, seed::Int)
    Random.seed!(seed)
    v = [0.1, 0.1]
    gas = Model([1, 12], [1, 12], D, scaling)

    gas.ω = v
    gas.A[1]  = convert(Matrix{Float64}, Diagonal(3*v))
    gas.A[12] = convert(Matrix{Float64}, Diagonal(-3*v))
    gas.B[1]  = convert(Matrix{Float64}, Diagonal(3*v))
    gas.B[12] = convert(Matrix{Float64}, Diagonal(-3*v))

    series, param = simulate_recursion(gas, 5000)

    return series
end

function test_coefficients_GAS_1_1(gas::Model{D, T}, ω::Vector{T}, A::Matrix{T}, B::Matrix{T};
                                    atol = 1e-1, rtol = 1e-1) where {D <: Distribution, T}
    @test gas.ω[1] ≈ ω[1] atol = atol rtol = rtol
    @test gas.ω[2] ≈ ω[2] atol = atol rtol = rtol
    @test gas.A[1][1, 1] ≈ A[1, 1] atol = atol rtol = rtol
    @test gas.A[1][2, 2] ≈ A[2, 2] atol = atol rtol = rtol
    @test gas.B[1][1, 1] ≈ B[1, 1] atol = atol rtol = rtol
    @test gas.B[1][2, 2] ≈ B[2, 2] atol = atol rtol = rtol
    return
end

function test_coefficients_GAS_1_12(gas::Model{D, T}; atol = 1e-1, rtol = 1e-1) where {D <: Distribution, T}
    @test gas.ω[1] ≈ 0.1 atol = atol rtol = rtol
    @test gas.ω[2] ≈ 0.1 atol = atol rtol = rtol
    @test gas.A[1][1, 1] ≈ 0.3 atol = atol rtol = rtol
    @test gas.A[1][2, 2] ≈ 0.3 atol = atol rtol = rtol
    @test gas.A[12][1, 1] ≈ -0.3 atol = atol rtol = rtol
    @test gas.A[12][2, 2] ≈ -0.3 atol = atol rtol = rtol
    @test gas.B[1][1, 1] ≈ 0.3 atol = atol rtol = rtol
    @test gas.B[1][2, 2] ≈ 0.3 atol = atol rtol = rtol
    @test gas.B[12][1, 1] ≈ -0.3 atol = atol rtol = rtol
    @test gas.B[12][2, 2] ≈ -0.3 atol = atol rtol = rtol
    return
end

function normality_quantile_and_pearson_residuals(D, n::Int, lags::Int; seed::Int = 11)
    Random.seed!(seed)

    gas = Model(1, 1, D, 0.0)
    gas.ω .= [0.0; 0.0]
    gas.A[1][[1; 4]] .= 0.2
    gas.B[1][[1; 4]] .= 0.2

    y, params = simulate_recursion(gas, n)
    quant_res = quantile_residuals(y, gas; initial_params = params[1:1, :])
    pearson = pearson_residuals(y, gas, initial_params = params[1:1, :])

    # quantile residuals
    jb = JarqueBeraTest(quant_res)
    @test pvalue(jb) >= 0.05
    Ljb = LjungBoxTest(quant_res, lags)
    @test pvalue(Ljb) >= 0.05

    # pearson
    Ljb = LjungBoxTest(pearson, lags)
    @test pvalue(Ljb) >= 0.05
    return
end

function test_GARCH_1_1(y, seed::Int, optimizer; atol = 1e-4, rtol = 1e-4)
    Random.seed!(seed)
    ini = [mean(y) var(y)]
    ub = [1.0; 1.0; 1.0; 1.0]
    lb = [-1.0; 0.0; 0.0; 0.0]
    gas = Model(1, 1, Normal, 1.0, time_varying_params = [2])
    f = fit!(gas, y; initial_params = ini, verbose = 2, opt_method = optimizer(gas, 10; ub = ub, lb = lb))
    show(stdout, fit_stats(f))

    @test gas.ω[1] - -0.00616637237701241 ≈ 0 atol = atol rtol = rtol
    @test gas.ω[2] - 0.010760592759725487 ≈ 0 atol = atol rtol = rtol
    @test gas.A[1][2, 2] - 0.15341133818189595 ≈ 0 atol = atol rtol = rtol
    @test gas.B[1][2, 2] - (0.15341133818189595 + 0.8058745318161223) ≈ 0 atol = atol rtol = rtol
    return
end
