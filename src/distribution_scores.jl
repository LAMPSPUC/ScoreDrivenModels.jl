# export score

"""
    score(y, dist::Distribution)


"""
function score(y, dist::Distribution, d::Float64)
    if d in (0.0, 0.5, 1)
        return score(y, dist)*fisher_information(dist)^d
    end
    return error("d must be a value in (0, 0.5, 1)")
end

function score(y, dist::Distribution)
    return error("score not implemented for $(typeof(dist)) distribution")
end

function fisher_information(dist::Distribution)
    return error("fisher information not implemented for $(typeof(dist)) distribution")
end

"""
Proof somewhere
"""
function score(y, poisson::Poisson)
    return (y - poisson.λ)/poisson.λ
end

"""
Proof somewhere
"""
function fisher_information(poisson::Poisson)
    return 1/poisson.λ
end

