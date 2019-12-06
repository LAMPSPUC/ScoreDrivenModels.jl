const SCORE_BIG_NUM = 1e5

mutable struct AuxiliaryStruct{T <: AbstractFloat}
    jac::Vector{T}
    fisher::Matrix{T}
    score_til_t::Vector{T}

    function AuxiliaryStruct{T}(n_pars::Integer) where T
        return new{T}(
            Vector{T}(undef, n_pars),
            Matrix{T}(undef, n_pars, n_pars),
            Vector{T}(undef, n_pars)
        )
    end
end

function score_tilde!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, 
                      param::Matrix{T}, aux::AuxiliaryStruct{T}, scaling::T, t::Int) where T

    if scaling == SCALINGS[1] # 0.0
        scaling_identity!(score_til, y, D, aux, param, t)
    elseif scaling == SCALINGS[2] # 1/2
        scaling_invsqrt!(score_til, y, D, aux, param, t)
    elseif scaling == SCALINGS[3] # 1
        scaling_inv!(score_til, y, D, aux, param, t)
    end

    NaN2zero!(score_til, t)
    big_threshold!(score_til, SCORE_BIG_NUM, t)
    small_threshold!(score_til, SMALL_NUM, t)
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

function scaling_invsqrt!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, 
                          aux::AuxiliaryStruct{T}, param::Matrix{T}, t::Int) where T
    # Evaluate jacobian, score and FI
    jacobian_link!(aux, D, param, t)
    score!(score_til, y, D, param, t)
    fisher_information!(aux, D, param, t)

    cholesky!(aux.fisher)
    for p in eachindex(aux.score_til_t)
        aux.score_til_t[p] = score_til[t, p]
    end
    aux.fisher = cholesky(aux.fisher).L
    aux.score_til_t = aux.fisher\aux.score_til_t
    for p in eachindex(aux.score_til_t)
        score_til[t, p] = aux.score_til_t[p]
    end
    return
end

function scaling_inv!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, 
                      aux::AuxiliaryStruct{T}, param::Matrix{T}, t::Int) where T
    # Evaluate jacobian, score and FI
    jacobian_link!(aux, D, param, t)
    score!(score_til, y, D, param, t)
    fisher_information!(aux, D, param, t)
    for p in eachindex(aux.score_til_t)
        aux.score_til_t[p] = score_til[t, p]
    end
    # Solve for inv(FI)*score_til
    LinearAlgebra.LAPACK.posv!('U', aux.fisher, aux.score_til_t)
    for p in eachindex(aux.jac)
        score_til[t, p] = aux.jac[p] * aux.score_til_t[p]
    end
    return
end