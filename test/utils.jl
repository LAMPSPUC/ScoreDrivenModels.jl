function test_score_mean(D::Type{<:Distribution}; n::Int = 10^7, seed::Int = 10,
                            atol::Float64 = 1e-3, rtol::Float64 = 1e-3)
    Random.seed!(seed)
    dist = D()
    pars = [params(dist)...]
    avg  = zeros(SDM.num_params(D))
    for i = 1:n
        avg += SDM.score(rand(D(pars...)), D, pars)
    end
    avg ./= n
    @test avg ≈ zeros(SDM.num_params(D)) atol = atol rtol = rtol
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

function test_dynamic(sigma::Float64, lags::Int; seed::Int = 12, atol::Float64 = 1e-1, rtol::Float64 = 1e-3, n::Int = 100)
    Random.seed!(seed)
    dist = Normal(0, sigma^2)
    obs_dynamic = kron(ones(n), collect(1:lags)) + rand(dist, n*lags)
    gas_lag_lag = GAS(lags, lags, Normal, 0.0)
    initial_params = dynamic_initial_params(obs_dynamic, gas_lag_lag)
    for i in eachindex(initial_params)
        @test initial_params[i][1] ≈ i atol = atol rtol = rtol
        @test initial_params[i][2] ≈ sigma atol = atol rtol = rtol
    end
end

function simulate_GAS_1_1(D::Type{<:Distribution}, scaling::Float64, ω::Vector{T}, A::Matrix{T}, 
                            B::Matrix{T}, seed::Int) where T
    Random.seed!(seed)
    gas = GAS(1, 1, D, scaling)

    gas.ω = ω
    gas.A[1] = A
    gas.B[1] = B
    series, param = simulate(gas, 5000)

    return series
end

function simulate_GAS_1_12(D::Type{<:Distribution}, scaling::Float64, seed::Int)
    Random.seed!(seed)
    v = [0.1, 0.1]
    gas = GAS([1, 12], [1, 12], D, scaling)

    gas.ω = v
    gas.A[1]  = convert(Matrix{Float64}, Diagonal(3*v))
    gas.A[12] = convert(Matrix{Float64}, Diagonal(-3*v))
    gas.B[1]  = convert(Matrix{Float64}, Diagonal(3*v))
    gas.B[12] = convert(Matrix{Float64}, Diagonal(-3*v))

    series, param = simulate(gas, 5000)

    return series
end

function test_coefficients_GAS_1_1(gas::GAS{D, T}, ω::Vector{T}, A::Matrix{T}, B::Matrix{T}; 
                                    atol = 1e-1, rtol = 1e-1) where {D <: Distribution, T}
    @test gas.ω[1] ≈ ω[1] atol = atol rtol = rtol
    @test gas.ω[2] ≈ ω[2] atol = atol rtol = rtol
    @test gas.A[1][1, 1] ≈ A[1, 1] atol = atol rtol = rtol
    @test gas.A[1][2, 2] ≈ A[2, 2] atol = atol rtol = rtol
    @test gas.B[1][1, 1] ≈ B[1, 1] atol = atol rtol = rtol
    @test gas.B[1][2, 2] ≈ B[2, 2] atol = atol rtol = rtol
    return
end

function test_coefficients_GAS_1_12(gas::GAS{D, T}; atol = 1e-1, rtol = 1e-1) where {D <: Distribution, T}
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

    gas = GAS(1, 1, D, 0.0)
    gas.ω .= [0.0; 0.0] 
    gas.A[1][[1; 4]] .= 0.2
    gas.B[1][[1; 4]] .= 0.2

    y, params = simulate(gas, n)
    quant_res = quantile_residuals(y, gas, [params[1]])
    pearson = pearson_residuals(y, gas, [params[1]])

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