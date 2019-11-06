abstract type ScoreDrivenModel{D, T} end

abstract type Unknowns_SDM end

"""
    AbstractOptimizationMethod
    
Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractOptimizationMethod{T} end

const SDM{D, T} = ScoreDrivenModel{D, T}