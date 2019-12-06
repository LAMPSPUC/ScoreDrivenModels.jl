struct AuxiliaryStruct{T <: AbstractFloat}
    jac::Vector{T}
    fisher::Matrix{T}

    function AuxiliaryStruct{T}(n_pars::Integer) where T
        return new{T}(
            Vector{T}(undef, n_pars),
            Matrix{T}(undef, n_pars, n_pars)
        )
    end
end

function score_tilde!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, 
                      param::Matrix{T}, aux::AuxiliaryStruct{T}, scaling::T, t::Int) where T

    if scaling == 0
        scaling_identity!(score_til, y, D, aux, param, t)
    elseif scaling == 1/2
        score_til = scaling_invsqrt(y, D, param)
    elseif scaling == 1.0
        score_til = scaling_inv(y, D, param)
    end

    NaN2zero!(score_til, t)
    big_threshold!(score_til, 1e5, t)
    small_threshold!(score_til, 1e-10, t)
    return
end

# Scalings
function scaling_identity!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, 
                           aux::AuxiliaryStruct{T}, param::Matrix{T}, t::Int) where T
    jacobian_link!(aux, D, param, t)
    score!(score_til, y, D, param, t)
    for p in eachindex(aux.jac)
        score_til[t, p] = score_til[t, p]/aux.jac[p]
    end
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