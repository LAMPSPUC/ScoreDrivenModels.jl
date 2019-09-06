# export score

"""
    score(y, dist::Distribution)


"""
function score(y, dist::Distribution, scaling)
    if scaling in (0.0, 0.5, 1)
        return fisher_information(dist, scaling)*score(y, dist)
    end
    return error("d must be a value in (0, 0.5, 1)")
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
function score(y, poisson::Poisson)
    return [(y - poisson.λ)/poisson.λ]
end

"""
Proof somewhere
"""
function fisher_information(poisson::Poisson, scaling)
    return (1/poisson.λ)^scaling
end

# Normal
"""
Proof somewhere
"""
function score(y, normal::Normal)
    return [
        (y - normal.μ)/normal.σ^2;
        (y - normal.μ)^2/normal.σ^3 - 1/normal.σ
    ]
end

"""
Proof somewhere
"""
function fisher_information(normal::Normal, scaling)
    return [
        1/(normal.σ^2) 0;
        0 2/(normal.σ^2)
    ]^2
end