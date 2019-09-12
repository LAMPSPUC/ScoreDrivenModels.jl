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
    # Allocations
    serie = zeros(n)
    param = Array{Vector{Float64}}(undef, n)
    param_tilde = Array{Vector{Float64}}(undef, n)

    # initial_values 
    dist = update_dist(sd_model.dist, initial_values)
    serie[1] = sample(dist)
    param_tilde[1] = param_to_param_tilde(dist, initial_values)
    
    for i in 1:n-1
        param[i] = param_tilde_to_param(sd_model.dist, param_tilde[i])
        param_tilde[i + 1] = sd_model.ω + sd_model.A*score_tilde(serie[i], dist, param[i], param_tilde[i], sd_model.scaling) + sd_model.B*param_tilde[i]
        updated_dist = update_dist(sd_model.dist, param_tilde_to_param(sd_model.dist, param_tilde[i + 1]))
        serie[i + 1] = sample(updated_dist)
    end
    return serie, param, param_tilde
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