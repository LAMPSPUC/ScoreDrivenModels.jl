function score_tilde(y::T, D::Type{<:Distribution}, param::Vector{T}, param_tilde::Vector{T}, scaling::T) where T
    @assert scaling in (0.0, 1/2, 1.0)

    if scaling == 0
        score_til = scaling_identity(y, D, param)
    elseif scaling == 1/2
        score_til = scaling_invsqrt(y, D, param)
    elseif scaling == 1.0
        score_til = scaling_inv(y, D, param)
    end

    NaN2zero!(score_til)
    big_threshold!(score_til, 1e5)
    small_threshold!(score_til, 1e-10)
    return score_til
end

# Scalings
function scaling_identity(y::T, D::Type{<:Distribution}, param::Vector{T}) where T
    jac = jacobian_link(D, param)
    return inv(jac)*score(y, D, param)
end

function scaling_invsqrt(y::T, D::Type{<:Distribution}, param::Vector{T}) where T
    jac = jacobian_link(D, param)
    inv_fisher = inv(fisher_information(D, param))
    J_tilde = cholesky(jac*inv_fisher*jac').L
    return J_tilde * inv(jac) * score(y, D, param)
end

function scaling_inv(y::T, D::Type{<:Distribution}, param::Vector{T}) where T
    jac = jacobian_link(D, param)
    return jac*inv(fisher_information(D, param))*score(y, D, param)
end