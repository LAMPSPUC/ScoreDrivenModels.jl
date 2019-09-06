export SDModel, simulate

mutable struct SDModel
    ω::Vector{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}
    dist::Distribution
    scaling

    function SDModel(ω, A, B, dist, scaling)
        return new(ω, A, B, dist, scaling)
    end
end

function simulate(sd_model::SDModel, n::Int, initial_values)
    serie = zeros(n)
    param = Array{Vector{Float64}}(undef, n)
    dist = update_dist(sd_model.dist, initial_values)
    serie[1] = sample(dist)
    param[1] = initial_values
    for i in 1:n-1
        param[i + 1] = sd_model.ω + sd_model.A*score(serie[i], update_dist(sd_model.dist, param[i]), sd_model.scaling) + sd_model.B*param[i]
        serie[i + 1] = sample(update_dist(sd_model.dist, param[i + 1]))
    end
    return serie, param
end

function update_dist(dist::Distribution, value)
    error("not implemented")
end 

function update_dist(dist::Poisson, value)
    return Poisson(value[1])
end 

function update_dist(dist::Normal, values)
    return Normal(values[1], values[2])
end 