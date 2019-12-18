abstract type ScoreDrivenModel{D, T} end

abstract type UnknownsSDM end

"""
    AbstractOptimizationMethod
    
Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractOptimizationMethod{T} end

const SDM{D, T} = ScoreDrivenModel{D, T}