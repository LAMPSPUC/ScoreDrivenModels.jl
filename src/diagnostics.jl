export quantile_residuals

function quantile_residuals(obs::Vector{T}, gas_sarima::GAS_Sarima{D, T}, initial_params::Vector{Vector{T}}) where {D, T}

    n = length(obs)
    len_ini_par = length(initial_params)
    quant_res = Vector{T}(undef, n - len_ini_par)
    params_fitted = score_driven_recursion(gas_sarima, obs, initial_params)
    

    for (i, param) in enumerate(params_fitted[len_ini_par + 1:end - 1])
        dist = D(param...)
        prob = cdf(dist, obs[i + len_ini_par])
        quant_res[i] = quantile(Normal(0, 1), prob)
    end

    return quant_res
end

function confidence_intervals(gas_sarima::GAS_Sarima, best_psi::Vector{T}; alpha::T = 0.05) where T
    dist = MvNormal(best_psi, best_psi)
    # Ask cristiano how to do this

end