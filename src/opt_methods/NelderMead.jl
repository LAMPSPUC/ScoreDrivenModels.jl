export NelderMead

mutable struct NelderMead{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    iterations::Int
    initial_points::Vector{Vector{T}}
end

"""
    NelderMead(model::Model, args...; kwargs...)

If an `Int` is provided the method will sample random initial_points and use them as initial 
points for Optim NelderMead method. If a `Vector{Vector{T}}` is provided it will use them as 
initial points for Optim NelderMead method.
"""
function NelderMead(model::Model{D, T}, n_initial_points::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                               iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6) where {D, T}

    initial_points = create_initial_points(model, n_initial_points, LB, UB)

    return NelderMead{T}(f_tol, g_tol, iterations, initial_points)
end

function NelderMead(model::Model{D, T}, initial_points::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                                           iterations::Int = 10^5) where {D, T}

    ensure_seeds_dimensions(model, initial_points)
    
    return NelderMead{T}(f_tol, g_tol, iterations, initial_points)
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::NelderMead{T}, verbose::Int, i::Int) where T
    return optimize(func, opt_method, Optim.NelderMead(), verbose, i)
end