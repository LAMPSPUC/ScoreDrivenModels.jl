# export score

"""
    score(y, dist::Distribution)
"""
function score_tilde(y, dist::Distribution, param::Vector{T}, param_tilde::Vector{T}, scaling::T) where T
    if scaling in (0.0, 0.5, 1)
        return fisher_information_tilde(dist, param, param_tilde, scaling)*score_tilde(y, dist, param, param_tilde)
    end
    return error("d must be a value in (0, 0.5, 1)")
end

function score_tilde(y, dist::Distribution)
    return error("score not implemented for $(typeof(dist)) distribution")
end
function fisher_information_tilde(dist::Distribution)
    return error("fisher information not implemented for $(typeof(dist)) distribution")
end
function score(y, dist::Distribution)
    return error("score not implemented for $(typeof(dist)) distribution")
end
function fisher_information(dist::Distribution)
    return error("fisher information not implemented for $(typeof(dist)) distribution")
end

# Poisson
"""
Proof somewhere
"""
function score(y, poisson::Poisson, param)
    return (y - param[1])/param[1]
end
function score_tilde(y, poisson::Poisson, param::Vector{T}, param_tilde::Vector{T}) where T
    return jacobian_param_tilde(poisson, param_tilde)*score(y, poisson, param)
end

"""
Proof somewhere
"""
function fisher_information(poisson::Poisson, param::Vector{T}, scaling::T) where T
    return (1/param[1])^scaling
end
function fisher_information_tilde(poisson::Poisson, param::Vector{T}, param_tilde::Vector{T}, scaling::T) where T
    return jacobian_param_tilde(poisson, param_tilde)'*fisher_information(poisson, param, scaling)*jacobian_param_tilde(poisson, param_tilde)
end

# # Normal
# """
# Proof somewhere
# """
# function score(y, normal::Normal)
#     return [
#         (y - normal.μ)/normal.σ^2;
#         (y - normal.μ)^2/normal.σ^3 - 1/normal.σ
#     ]
# end

# """
# Proof somewhere
# """
# function fisher_information(normal::Normal, scaling)
#     return [
#         1/(normal.σ^2) 0;
#         0 2/(normal.σ^2)
#     ]^scaling
# end