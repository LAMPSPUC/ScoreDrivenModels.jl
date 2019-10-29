export quantile_residuals, pearson_residuals

function quantile_residuals(obs::Vector{T}, gas_sarima::GAS_Sarima{D, T}, initial_params::Vector{Vector{T}}) where {D, T}

    n = length(obs)
    len_ini_par = length(initial_params)
    quant_res = Vector{T}(undef, n - len_ini_par)
    params_fitted = score_driven_recursion(gas_sarima, obs, initial_params)

    for (i, param) in enumerate(params_fitted[len_ini_par + 1:end - 1])
        dist = update_dist(D, param)
        prob = cdf(dist, obs[i + len_ini_par])
        quant_res[i] = quantile(Normal(0, 1), prob)

        # Treat possible infinity values
        if quant_res[i] == Inf
            quant_res[i] = 1e5
        elseif quant_res[i] == -Inf
            quant_res[i] = -1e5
        end
    end

    return quant_res
end

function pearson_residuals(obs::Vector{T}, gas_sarima::GAS_Sarima{D, T}, initial_params::Vector{Vector{T}}) where {D, T}

    n = length(obs)
    len_ini_par = length(initial_params)
    pearson = Vector{T}(undef, n - len_ini_par)
    params_fitted = score_driven_recursion(gas_sarima, obs, initial_params)

    for (i, param) in enumerate(params_fitted[len_ini_par + 1:end - 1])
        dist = update_dist(D, param)
        pearson[i] = (obs[i + len_ini_par] - mean(dist))/sqrt(var(dist))
    end

    return pearson
end