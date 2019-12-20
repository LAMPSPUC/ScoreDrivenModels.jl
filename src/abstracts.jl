export AuxiliaryLinAlg

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

export GAS

mutable struct GAS{D <: Distribution, T <: AbstractFloat} <: SDM{D, T}
    ω::Vector{T}
    A::Dict{Int, Matrix{T}}
    B::Dict{Int, Matrix{T}}
    scaling::Real
    links::Vector{<:Link}
end