export simulate

function simulate(sd_model::SDModel, n::Int)
    return simulate(sd_model, n, stationary_initial_params(sd_model))
end

function simulate(sd_model::SDModel, n::Int, initial_param::Vector{T}) where T
    # Allocations
    serie = zeros(n)
    param = Vector{Vector{T}}(undef, n)
    param_tilde = Vector{Vector{T}}(undef, n)

    # initial_values 
    dist = update_dist(sd_model.dist, param_tilde_to_param(sd_model.dist, initial_param))
    serie[1] = sample_observation(dist)
    param_tilde[1] = initial_param

    for i in 1:n-1
        # update step
        univariate_score_driven_update!(param, param_tilde, serie[i], sd_model, i)
        # Sample from the updated distribution
        updated_dist = update_dist(sd_model.dist, param_tilde_to_param(sd_model.dist, param_tilde[i + 1]))
        serie[i + 1] = sample_observation(updated_dist)
    end
    update_param!(param, param_tilde, sd_model.dist, n #=end=#)

    return serie, param
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

function update_dist(dist::Beta, param::Vector{T}) where T
    return Beta(param[1], param[2])
end 