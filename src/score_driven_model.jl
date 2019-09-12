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
        score_til = score_tilde(serie[i], dist, param[i], param_tilde[i], sd_model.scaling)
        update_param_tilde(param_tilde, dist, sd_model.ω, sd_model.A, sd_model.B, score_til, i)
        updated_dist = update_dist(sd_model.dist, param_tilde_to_param(sd_model.dist, param_tilde[i + 1]))
        serie[i + 1] = sample_observation(updated_dist)
    end
    param[end] = param_tilde_to_param(sd_model.dist, param_tilde[end])
    return serie, param, param_tilde
end

function update_param_tilde(param_tilde, dist, ω, A, B, score_til, i)
    param_tilde[i + 1] = ω + A*score_til + B*param_tilde[i]
    return 
end

function update_dist(dist::Distribution, param::Vector{T}) where T
    error("not implemented")
end 

function update_dist(dist::Poisson, param::Vector{T}) where T
    return Poisson(param[1])
end 

function update_dist(dist::Normal, param::Vector{T}) where T
    return Normal(param[1], param[2])
end 