function normality_quantile_and_pearson_residuals(D, n::Int, lags::Int; seed::Int = 10)
    Random.seed!(seed)

    gas_sarima = GAS_Sarima(1, 1, D, 0.0)
    gas_sarima.ω .= [0.0; 0.0] 
    gas_sarima.A[1][[1; 4]] .= 0.2
    gas_sarima.B[1][[1; 4]] .= 0.2

    y, params = simulate(gas_sarima, n)
    quant_res = quantile_residuals(y, gas_sarima, [params[1]])
    pearson = pearson_residuals(y, gas_sarima, [params[1]])

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