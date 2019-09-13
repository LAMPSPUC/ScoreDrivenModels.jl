function score_tilde(y, dist::Distribution, param::Vector{T}, param_tilde::Vector{T}, scaling::T) where T
    @assert scaling in (0.0, 1/2, 1.0)
    # Evaluate Jacobian and score tilde (\tilde \nabla)
    jac = jacobian_param_tilde(dist, param_tilde)
    score_til = jac*score(y, dist, param)
    if scaling == 0.0
        return score_til
    elseif scaling == 1/2
        return scaling_invsqrt(jac, dist, param)*score_til
    elseif scaling == 1.0
        return scaling_inv(jac, dist, param)*score_til
    end
end

function score_tilde_threshold(y, dist::Distribution, param::Vector{T}, param_tilde::Vector{T}, scaling::T, threshold) where T
    # add some treatment to NaN and big numbers
end

function score(y, dist::Distribution)
    return error("score not implemented for $(typeof(dist)) distribution")
end
function fisher_information(dist::Distribution)
    return error("fisher information not implemented for $(typeof(dist)) distribution")
end

# Scalings
function scaling_invsqrt(jac, dist, param)
    #TODO improve performace and check if this is right
    chol = cholesky(fisher_information(dist, param))
    cholmat = Matrix(chol)
    return jac'*inv(cholmat)*jac
end

function scaling_inv(jac, dist, param)
    #TODO improve performace
    return jac'*inv(fisher_information(dist, param))*jac
end