# export score

"""
    score(y, dist::Distribution)
"""
function score_tilde(y, dist::Distribution, param::Vector{T}, param_tilde::Vector{T}, scaling::T) where T
    @assert scaling in (0.0, 1/2, 1.0)
    return (fisher_information_tilde(dist, param, param_tilde)^scaling)*score_tilde(y, dist, param, param_tilde)
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

# Transformations 
function score_tilde(y, dist::Distribution, param::Vector{T}, param_tilde::Vector{T}) where T
    return jacobian_param_tilde(dist, param_tilde).*score(y, dist, param)
end
function fisher_information_tilde(dist::Distribution, param::Vector{T}, param_tilde::Vector{T}) where T
    return jacobian_param_tilde(dist, param_tilde)'*fisher_information(dist, param)*jacobian_param_tilde(dist, param_tilde)
end

# Poisson
"""
Proof somewhere
"""
function score(y, poisson::Poisson, param)
    return (y - param[1])/param[1]
end

"""
Proof somewhere
"""
function fisher_information(poisson::Poisson, param::Vector{T}) where T
    return (1/param[1])
end

# Normal
"""
Proof somewhere
"""
function score(y, normal::Normal, param::Vector{T}) where T
    return [
        (y - param[1])/param[2]^2;
        (y - param[1])^2/param[2]^3 - 1/param[2]
    ]
end

"""
Proof somewhere
"""
function fisher_information(normal::Normal, param::Vector{T}) where T
    return [
        1/(param[2]^2) 0;
        0 2/(param[2]^2)
    ]
end