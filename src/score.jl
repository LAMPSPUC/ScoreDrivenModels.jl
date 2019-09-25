function score_tilde(y, D::Type{<:Distribution}, param::Vector{T}, param_tilde::Vector{T}, scaling::T) where T
    @assert scaling in (0.0, 1/2, 1.0)
    # Evaluate Jacobian and score tilde (\tilde \nabla)
    jac = jacobian_param_tilde(D, param_tilde)
    score_til = jac*score(y, D, param)
    # if scaling == 0 do nothing
    if scaling == 1/2
        score_til = scaling_invsqrt(jac, D, param)*score_til
    elseif scaling == 1.0
        score_til = scaling_inv(jac, D, param)*score_til
    end

    NaN2zero!(score_til)
    big_threshold!(score_til, 1e5)
    small_threshold!(score_til, 1e-10)
    return score_til
end

function score(y, D::Type{<:Distribution})
    return error("score not implemented for $(typeof(dist)) distribution")
end
function fisher_information(D::Type{<:Distribution})
    return error("fisher information not implemented for $(typeof(dist)) distribution")
end

# Scalings
function scaling_invsqrt(jac, dist, param)
    #TODO improve performace and check if this is right
    # chol = cholesky(fisher_information(dist, param))
    # cholmat = Matrix(chol)
    error("check how to do this properly")
    return jac'*inv(cholmat)*jac
end

function scaling_inv(jac, dist, param)
    #TODO improve performace
    return jac'*inv(fisher_information(dist, param))*jac
end