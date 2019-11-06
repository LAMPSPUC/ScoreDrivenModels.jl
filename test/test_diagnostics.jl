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

@testset "quantile and pearson residuals" begin
    n = 1000
    lags = 1
    normality_quantile_and_pearson_residuals(Beta, n, lags)
    normality_quantile_and_pearson_residuals(Normal, n, lags)
    normality_quantile_and_pearson_residuals(LogNormal, n, lags)
    normality_quantile_and_pearson_residuals(Gamma, n, lags)
end