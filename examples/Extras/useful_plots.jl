# Array of distributions
function array_of_distributions(::Normal, params::Vector{Vector{T}}) where T
    dists = Array{Normal{T}}(undef, length(params))
    for i in 1:length(params)
        dists[i] = Normal(param[i][1], param[i][2])
    end
    return dists
end

function plot_sdm(obs::Vector{T}, dists; quantiles::Vector{T} = [0.05; 0.95]) where T
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

function plot_sdm(obs::Vector{T}, params::Vector{Vector{T}}, dist::Distribution; quantiles::Vector{T} = [0.05; 0.95]) where T
    dists = array_of_distributions(dist, params)
    return plot_sdm(obs, dists; quantiles = quantiles)
end
