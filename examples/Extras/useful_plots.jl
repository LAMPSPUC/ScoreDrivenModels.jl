using ScoreDrivenModels

# Array of distributions
function array_of_distributions(dist::Distribution, params::Vector{Vector{T}}) where T
    dists = Array{typeof(dist)}(undef, length(params))
    for i in 1:length(params)
        dists[i] = ScoreDrivenModels.update_dist(dist, params[i])
    end
    return dists
end

# Utils
function transform_to_vecofvec(param::Vector{Vector{Float64}})
    return param
end

function transform_to_vecofvec(param::Vector{Float64})
    vparam = Vector{Vector{Float64}}(undef, length(param))
    for i in eachindex(param)
        vparam[i] = [param[i]]
    end
    return vparam
end

function transform_to_vecofvec(param::Matrix{Float64})
    vparam = Vector{Vector{Float64}}(undef, size(param, 1))
    for i in 1:size(param, 1)
        vparam[i] = param[i, :]
    end
    return vparam
end

function plot_sdm(obs::Vector{T}, dists::Array; quantiles::Vector{T} = [0.05; 0.95]) where T
    p1 = plot(obs, label = "observations", color = "black")

    @assert length(quantiles) == 2
    @assert (quantiles[1] >= 0) && (quantiles[1] < quantiles[2]) && (quantiles[2] <= 1.0)

    len = length(obs)
    mean_vec = Array{T}(undef, len)
    upper_quantile = Array{T}(undef, len)
    lower_quantile = Array{T}(undef, len)

    for i in 1:len
        quant = quantile.(dists[i], quantiles)
        mean_vec[i] = mean(dists[i])
        upper_quantile[i] = quant[2]
        lower_quantile[i] = quant[1]
    end

    plot!(p1, mean_vec, color = "Steel Blue", label = "Confidence interval", fillrange = (lower_quantile, upper_quantile), fillalpha = 0.3)
    return p1
end

function plot_sdm(p1, obs::Vector{T}, dists::Array; quantiles::Vector{T} = [0.05; 0.95]) where T
    plot!(p1, obs, label = "observations 2", color = "grey")

    @assert length(quantiles) == 2
    @assert (quantiles[1] >= 0) && (quantiles[1] < quantiles[2]) && (quantiles[2] <= 1.0)

    len = length(obs)
    mean_vec = Array{T}(undef, len)
    upper_quantile = Array{T}(undef, len)
    lower_quantile = Array{T}(undef, len)

    for i in 1:len
        quant = quantile.(dists[i], quantiles)
        mean_vec[i] = mean(dists[i])
        upper_quantile[i] = quant[2]
        lower_quantile[i] = quant[1]
    end

    plot!(p1, mean_vec, color = "Indian Red", label = "Confidence interval 2", fillrange = (lower_quantile, upper_quantile), fillalpha = 0.3)
    return p1
end

function plot_sdm(obs::Vector{T}, params::Array, dist::Distribution; quantiles::Vector{T} = [0.05; 0.95]) where T
    dists = array_of_distributions(dist, transform_to_vecofvec(params))
    return plot_sdm(obs, dists; quantiles = quantiles)
end

function plot_sdm!(p1, obs::Vector{T}, params::Array, dist::Distribution; quantiles::Vector{T} = [0.05; 0.95]) where T
    dists = array_of_distributions(dist, transform_to_vecofvec(params))
    return plot_sdm(p1, obs, dists; quantiles = quantiles)
end