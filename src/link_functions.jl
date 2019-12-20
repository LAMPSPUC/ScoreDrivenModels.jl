export IdentityLink, LogLink, LogitLink
export link, unlink, jacobian_link

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

link(::IdentityLink, param::T) where T = param
unlink(::IdentityLink, param_tilde::T) where T = param_tilde
jacobian_link(::IdentityLink, param::T) where T = one(T)

"""
    LogLink <: Link

Define the map ``\\tilde{f} = \\ln(f - a)`` where ``f \\in [a, \\infty), a \\in \\mathbb{R}`` and ``\\tilde{f} \\in \\mathbb{R}``
"""
struct LogLink{T <: Real} <: Link 
    lb::T # Lower bound
end

link(log_link::LogLink{T}, param::T) where T = log(param - log_link.lb)
unlink(log_link::LogLink{T}, param_tilde::T) where T = exp(param_tilde) + log_link.lb
jacobian_link(log_link::LogLink{T}, param::T) where T = 1/(param - log_link.lb)

"""
    LogitLink <: Link

Define the map ``\\tilde{f} = \\-ln(\\frac{b - a}{f + a} - 1)`` where ``f \\in [a, b], a, b \\in \\mathbb{R}`` and ``\\tilde{f} \\in \\mathbb{R}``
"""
struct LogitLink{T <: Real} <: Link 
    lb::T
    ub::T
end

function link(logit_link::LogitLink, param::T) where T 
    return log((param - logit_link.lb)/(logit_link.ub - param))
end
function unlink(logit_link::LogitLink, param_tilde::T) where T 
    return logit_link.lb + ((logit_link.ub - logit_link.lb)/(1 + exp(-param_tilde)))
end
function jacobian_link(logit_link::LogitLink, param::T) where T 
    return (logit_link.ub + logit_link.lb)/((logit_link.ub - param) * (param - logit_link.lb))
end

const LINKS = [
    IdentityLink;
    LogLink;
    LogitLink
]
