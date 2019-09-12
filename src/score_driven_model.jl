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

function simulate(sd_model::SDModel, n::Int, initial_param::Vector{T}) where T
    # Allocations
    serie = zeros(n)
    param = Vector{Vector{T}}(undef, n)
    param_tilde = Vector{Vector{T}}(undef, n)

    # initial_values 
    dist = update_dist(sd_model.dist, initial_param)
    serie[1] = sample_observation(dist)
    param_tilde[1] = param_to_param_tilde(dist, initial_param)

    for i in 1:n-1
        param[i] = param_tilde_to_param(sd_model.dist, param_tilde[i])
        @show sd_model.A
        @show sd_model.B
        @show param_tilde[i]
        @show param[i]
        @show score_tilde(serie[i], dist, param[i], param_tilde[i], sd_model.scaling)
        param_tilde[i + 1] = sd_model.ω + sd_model.A*score_tilde(serie[i], dist, param[i], param_tilde[i], sd_model.scaling) + sd_model.B*param_tilde[i]
        updated_dist = update_dist(sd_model.dist, param_tilde_to_param(sd_model.dist, param_tilde[i + 1]))
        serie[i + 1] = sample_observation(updated_dist)
    end
    return serie, param, param_tilde
end

function update_dist(dist::Distribution, param::Vector{T}) where T
    error("not implemented")
end 

function update_dist(dist::Poisson, param::Vector{T}) where T
    return Poisson(param[1])
end 

function update_dist(dist::Normal, values)
    return Normal(values[1], values[2])
end 