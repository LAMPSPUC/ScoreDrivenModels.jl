@testset "quantile and pearson residuals" begin
    n = 10^3
    lags = 1
    normality_quantile_and_pearson_residuals(Normal, n, lags)
    normality_quantile_and_pearson_residuals(LogNormal, n, lags)
    normality_quantile_and_pearson_residuals(Gamma, n, lags)
end