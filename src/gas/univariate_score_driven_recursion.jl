export score_driven_recursion, fitted_mean

"""
score_driven_recursion(sd_model::SDM, observations::Vector{T}) where T

start with the stationary params for a 
"""
function score_driven_recursion(gas::GAS{D, T}, observations::Vector{T}) where {D, T}
    initial_params = stationary_initial_params(gas)
    return score_driven_recursion(gas, observations, initial_params)
end

function score_driven_recursion(gas::GAS{D, T}, observations::Vector{T}, initial_param::Matrix{T}) where {D, T}
    @assert gas.scaling in SCALINGS
    # Allocations
    n = length(observations)
    n_params = num_params(D)
    param = Matrix{T}(undef, n + 1, n_params)
    param_tilde = Matrix{T}(undef, n + 1, n_params)
    scores_tilde = Matrix{T}(undef, n, n_params)
    aux = AuxiliaryStruct{T}(n_params)

    # Query the biggest lag
    biggest_lag = number_of_lags(gas)

    # initial_values  
    for i in 1:biggest_lag
        for p in 1:n_params
            param[i, p] = initial_param[i, p]
        end
        link!(param_tilde, D, param, i)
        score_tilde!(scores_tilde, observations[i], D, param, aux, gas.scaling, i)
    end
    
    update_param_tilde!(param_tilde, gas.ω, gas.A, gas.B, scores_tilde, biggest_lag)

    for i in biggest_lag + 1:n
        univariate_score_driven_update!(param, param_tilde, scores_tilde, observations[i], aux, gas, i)
    end
    update_param!(param, param_tilde, D, n + 1)

    return param
end

function univariate_score_driven_update!(param::Matrix{T}, param_tilde::Matrix{T},
                                         scores_tilde::Matrix{T},
                                         observation::T, aux::AuxiliaryStruct{T},
                                         gas::GAS{D, T}, i::Int) where {D <: Distribution, T <: AbstractFloat}
    # update param 
    update_param!(param, param_tilde, D, i)
    # evaluate score
    score_tilde!(scores_tilde, observation, D, param, aux, gas.scaling, i)
    # update param_tilde
    update_param_tilde!(param_tilde, gas.ω, gas.A, gas.B, scores_tilde, i)
    return 
end

function update_param!(param::Matrix{T}, param_tilde::Matrix{T}, D::Type{<:Distribution}, i::Int) where T
    unlink!(param, D, param_tilde, i)
    # Some treatments 
    NaN2zero!(param, i)
    big_threshold!(param, BIG_NUM, i)
    small_threshold!(param, SMALL_NUM, i)
    return
end

function update_param_tilde!(param_tilde::Matrix{T}, ω::Vector{T}, A::Dict{Int, Matrix{T}}, 
                             B::Dict{Int, Matrix{T}}, scores_tilde::Matrix{T}, i::Int) where T
    for p in eachindex(ω)
        param_tilde[i + 1, p] = ω[p]
    end
    for (lag, mat) in A
        for p in axes(mat, 1)
            param_tilde[i + 1, p] += mat[p, p] * scores_tilde[i - lag + 1, p]
        end
    end
    for (lag, mat) in B
        for p in axes(mat, 1)
            param_tilde[i + 1, p] += mat[p, p] * param_tilde[i - lag + 1, p]
        end
    end
    return 
end

"""
    fitted_mean(gas::GAS{D, T}, observations::Vector{T}, initial_params::Vector{T}) where {D, T}

return the fitted mean.. #TODO
"""
function fitted_mean(gas::GAS{D, T}, observations::Vector{T}, initial_params::Vector{Vector{T}}) where {D, T}
    params_fitted = score_driven_recursion(gas, observations, initial_params)
    n = length(params_fitted)
    fitted_mean = Vector{T}(undef, n)

    for (i, param) in enumerate(params_fitted)
        sdm_dist = update_dist(D, param)
        fitted_mean[i] = mean(sdm_dist)
    end
    
    return fitted_mean
end