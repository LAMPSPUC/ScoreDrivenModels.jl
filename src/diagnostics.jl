export quantile_residuals, pearson_residuals

function quantile_residuals(obs::Vector{T}, gas::ScoreDrivenModel{D, T};
                            initial_params::Matrix{T} = stationary_initial_params(gas)) where {D, T}

    n = length(obs)
    len_ini_par = length(initial_params)
    quant_res = Vector{T}(undef, n - len_ini_par)
    params_fitted = score_driven_recursion(gas, obs; initial_params = initial_params)

    for t in axes(params_fitted[len_ini_par + 1:end - 1, :], 1)
        dist = update_dist(D, params_fitted[len_ini_par + 1:end - 1, :], t)
        # Continuous distributions PIT
        if supertype(D).parameters[2] == Continuous
            prob = cdf(dist, obs[t + len_ini_par])
            quant_res[t] = quantile(Normal(0, 1), prob)
        else # Discrete distributions PIT
            lb = cdf(dist, obs[t + len_ini_par] - 1)
            ub = cdf(dist, obs[t + len_ini_par])
            prob = rand(Uniform(lb, ub))
            quant_res[t] = quantile(Normal(0, 1), prob)
        end

        # Treat possible infinity values
        if quant_res[t] == Inf
            quant_res[t] = 1e5
        elseif quant_res[t] == -Inf
            quant_res[t] = -1e5
        end
    end

    return quant_res
end

function pearson_residuals(obs::Vector{T}, gas::ScoreDrivenModel{D, T};
                             initial_params::Matrix{T} = stationary_initial_params(gas)) where {D, T}

    n = length(obs)
    len_ini_par = length(initial_params)
    pearson = Vector{T}(undef, n - len_ini_par)
    params_fitted = score_driven_recursion(gas, obs; initial_params = initial_params)

    for t in axes(params_fitted[len_ini_par + 1:end - 1, :], 1)
        dist = update_dist(D, params_fitted[len_ini_par + 1:end - 1, :], t)
        pearson[t] = (obs[t + len_ini_par] - mean(dist))/sqrt(var(dist))
    end

    return pearson
end
