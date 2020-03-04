export IdentityLink, LogLink, LogitLink
export link, unlink, jacobian_link

abstract type Link end

"""
    link(args...)

The link function is a map that brings a parameter ``f`` in a subspace ``\\mathcal{F} \\subset \\mathbb{R}`` to ``\\mathbb{R}``.
"""
function link end

"""
    unlink(args...)

The unlink function is the inverse map of [`link`](@ref link). It brings ``\\tilde{f}`` in ``\\mathbb{R}`` to the subspace ``\\mathcal{F} \\subset \\mathbb{R}``.
"""
function unlink end

"""
    jacobian_link(args...)

Evaluates the derivative of the [`link`](@ref link) with respect to the parameter ``f``.
"""
function jacobian_link end

"""
    IdentityLink <: Link

Define the map ``\\tilde{f} = f`` where ``f \\in \\mathbb{R}`` and ``\\tilde{f} \\in \\mathbb{R}``
"""
struct IdentityLink <: Link end

link(::Type{IdentityLink}, param::T) where T = param
unlink(::Type{IdentityLink}, param_tilde::T) where T = param_tilde
jacobian_link(::Type{IdentityLink}, param::T) where T = one(T)

"""
    LogLink <: Link

Define the map ``\\tilde{f} = \\ln(f - a)`` where ``f \\in [a, \\infty), a \\in \\mathbb{R}`` and ``\\tilde{f} \\in \\mathbb{R}``
"""
struct LogLink <: Link end

link(::Type{LogLink}, param::T, lower_bound::T) where T = log(param - lower_bound)
unlink(::Type{LogLink}, param_tilde::T, lower_bound::T) where T = exp(param_tilde) + lower_bound
jacobian_link(::Type{LogLink}, param::T, lower_bound::T) where T = 1/(param - lower_bound)

"""
    LogitLink <: Link

Define the map ``\\tilde{f} = \\ln(\\frac{f - a}{b - f})`` where ``f \\in [a, b], a, b \\in \\mathbb{R}`` and ``\\tilde{f} \\in \\mathbb{R}``
"""
struct LogitLink <: Link end

function link(::Type{LogitLink}, param::T, lower_bound::T, upper_bound::T) where T 
    return log((param - lower_bound)/(upper_bound - param))
end
function unlink(::Type{LogitLink}, param_tilde::T, lower_bound::T, upper_bound::T) where T 
    return lower_bound + ((upper_bound - lower_bound)/(1 + exp(-param_tilde)))
end
function jacobian_link(::Type{LogitLink}, param::T, lower_bound::T, upper_bound::T) where T 
    return (upper_bound + lower_bound)/((upper_bound - param) * (param - lower_bound))
end

const LINKS = [
    IdentityLink;
    LogLink;
    LogitLink
]
