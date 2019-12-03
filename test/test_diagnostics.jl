@testset "quantile and pearson residuals" begin
    n = 1000
    lags = 1
    normality_quantile_and_pearson_residuals(Beta, n, lags)
    normality_quantile_and_pearson_residuals(Normal, n, lags)
    normality_quantile_and_pearson_residuals(LogNormal, n, lags)
    normality_quantile_and_pearson_residuals(Gamma, n, lags)
end