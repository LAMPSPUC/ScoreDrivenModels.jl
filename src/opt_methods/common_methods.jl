function ensure_seeds_dimensions(model::Model{D, T}, initial_points::Vector{Vector{T}}) where {D, T}
    # Querying number of unknowns
    n_psi = dim_unknowns(model)

    n_initial_points = length(initial_points)

    for (i, initial_point) in enumerate(initial_points)
        if length(initial_point) != n_psi
            error("Seed $i has $(length(initial_point)) elements and the model has $n_psi unknowns.")
        end
    end
    return 
end

function create_initial_points(model::Model{D, T}, n_initial_points::Int, LB::T, UB::T) where {D, T}
    # Querying number of unknowns
    n_psi = dim_unknowns(model)

    initial_points = Vector{Vector{T}}(undef, n_initial_points)

    # Generate initial values in [-1e3, 1e3]
    for i = 1:n_initial_points
        initial_points[i] = rand(Distributions.Uniform(LB, UB), n_psi)
    end
    return initial_points
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::AbstractOptimizationMethod{T},
                  optimizer::Optim.AbstractOptimizer, verbose::Int, i::Int, time_limit_sec::Int) where T
                  
    return Optim.optimize(func, opt_method.initial_points[i], optimizer,
                                                     Optim.Options(f_tol = opt_method.f_tol, 
                                                                   g_tol = opt_method.g_tol, 
                                                                   iterations = opt_method.iterations,
                                                                   show_trace = show_trace(verbose),
                                                                   time_limit = time_limit_sec))
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::AbstractOptimizationMethod{T},
                  cons::Optim.TwiceDifferentiableConstraints,
                  optimizer::Optim.AbstractOptimizer, 
                  verbose::Int, i::Int, time_limit_sec::Int) where T

    return Optim.optimize(func, cons, opt_method.initial_points[i], optimizer,
                                                     Optim.Options(f_tol = opt_method.f_tol, 
                                                                   g_tol = opt_method.g_tol, 
                                                                   iterations = opt_method.iterations,
                                                                   show_trace = show_trace(verbose), 
                                                                   time_limit = time_limit_sec))
end

show_trace(verbose::Int) =  (verbose == 3 ? true : false)