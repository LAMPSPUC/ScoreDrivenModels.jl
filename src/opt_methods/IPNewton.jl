export IPNewton

mutable struct IPNewton{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    ub::Vector{T}
    lb::Vector{T}
    iterations::Int
    initial_points::Vector{Vector{T}}
end

"""
    IPNewton(model::ScoreDrivenModel, args...; kwargs...)

If an `Int` is provided the method will sample that many random initial_points and use them as
initial points for Optim IPNewton method. If a `Vector{Vector{T}}` is provided it will use them as
initial points for Optim IPNewton method.

This method provides an alternative to create box constraints. constraints can be passed as a Vector
the default constraints are `ub = Inf * ones(T, dim_unknowns(model))` and `lb = -Inf * ones(T, dim_unknowns(model))`
"""
function IPNewton(model::ScoreDrivenModel{D, T}, n_initial_points::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6),
                                                ub::Vector{T} = Inf * ones(T, dim_unknowns(model)),
                                                lb::Vector{T} = -Inf * ones(T, dim_unknowns(model)),
                                                iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6) where {D, T}

    initial_points = create_initial_points(model, n_initial_points, LB, UB)

    return IPNewton{T}(f_tol, g_tol, ub, lb, iterations, initial_points)
end

function IPNewton(model::ScoreDrivenModel{D, T}, initial_points::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6),
                                                            ub::Vector{T} = Inf*ones(T, length(initial_points[1])),
                                                            lb::Vector{T} = -Inf*ones(T, length(initial_points[1])),
                                                            iterations::Int = 10^5) where {D, T}

    ensure_seeds_dimensions(model, initial_points)

    return IPNewton{T}(f_tol, g_tol, ub, lb, iterations, initial_points)
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::IPNewton{T}, verbose::Int, i::Int, time_limit_sec::Int) where T
    cons = Optim.TwiceDifferentiableConstraints(opt_method.lb, opt_method.ub)
    return optimize(func, opt_method, cons, Optim.IPNewton(), verbose, i, time_limit_sec)
end
