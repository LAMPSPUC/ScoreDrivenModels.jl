export IPNewton

mutable struct IPNewton{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    ub::Vector{T}
    lb::Vector{T}
    iterations::Int
    seeds::Vector{Vector{T}}
end

"""
    IPNewton(model::Model, args...; kwargs...)

If an `Int` is provided the method will sample random seeds and use them as initial points for Optim IPNewton method.
If a `Vector{Vector{T}}` is provided it will use them as initial points for Optim IPNewton method.
"""
function IPNewton(model::Model{D, T}, n_seeds::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                ub::Vector{T} = Inf*ones(T, dim_unknowns(model)),
                                                lb::Vector{T} = -Inf*ones(T, dim_unknowns(model)),
                                                iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6) where {D, T}

    seeds = create_seeds(model, n_seeds, LB, UB)

    return IPNewton{T}(f_tol, g_tol, ub, lb, iterations, seeds)
end

function IPNewton(model::Model{D, T}, seeds::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                            ub::Vector{T} = Inf*ones(T, length(seeds[1])),
                                                            lb::Vector{T} = -Inf*ones(T, length(seeds[1])),
                                                            iterations::Int = 10^5) where {D, T}

    ensure_seeds_dimensions(model, seeds)
    
    return IPNewton{T}(f_tol, g_tol, ub, lb, iterations, seeds)
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::IPNewton{T}, verbose::Int, i::Int) where T
    cons = Optim.TwiceDifferentiableConstraints(opt_method.lb, opt_method.ub)
    return optimize(func, opt_method, cons, Optim.IPNewton(), verbose, i)
end