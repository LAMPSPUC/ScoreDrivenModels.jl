export AuxiliaryLinAlg, GAS

abstract type ScoreDrivenModel{D, T} end

abstract type UnknownsSDM end

"""
    AbstractOptimizationMethod
    
Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractOptimizationMethod{T} end

const SDM{D, T} = ScoreDrivenModel{D, T}

abstract type Link end

mutable struct AuxiliaryLinAlg{T <: AbstractFloat}
    jac::Vector{T}
    fisher::Matrix{T}
    score_til_t::Vector{T}

    function AuxiliaryLinAlg{T}(n_pars::Integer) where T
        return new{T}(
            Vector{T}(undef, n_pars),
            Matrix{T}(undef, n_pars, n_pars),
            Vector{T}(undef, n_pars)
        )
    end
end

mutable struct GAS{D <: Distribution, T <: AbstractFloat} <: SDM{D, T}
    ω::Vector{T}
    A::Dict{Int, Matrix{T}}
    B::Dict{Int, Matrix{T}}
    scaling::Real
    links::Vector{<:Link}
end

struct FittedSDM{T <: AbstractFloat}
    aic::T
    bic::T
    llk::T
    coefs::Vector{T}
    numerical_hessian::Matrix{T}
end

mutable struct AuxEstimation{T <: AbstractFloat}
    psi::Vector{Vector{T}}
    numerical_hessian::Vector{Matrix{T}}
    loglikelihood::Vector{T}
    opt_result::Vector{Optim.OptimizationResults}

    function AuxEstimation{T}() where T
        return new(
            Vector{Vector{T}}(undef, 0), #psi
            Vector{Matrix{T}}(undef, 0), 
            Vector{T}(undef, 0), # loglikelihood
            Vector{Optim.OptimizationResults}(undef, 0) # opt_result
            )
    end
end
