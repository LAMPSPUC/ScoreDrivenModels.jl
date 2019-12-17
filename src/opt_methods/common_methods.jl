function ensure_seeds_dimensions(model::SDM{D, T}, seeds::Vector{Vector{T}}) where {D, T}
    # Querying number of unknowns
    n_psi = dim_unknowns(model)

    n_seeds = length(seeds)

    for (i, seed) in enumerate(seeds)
        if length(seed) != n_psi
            error("Seed $i has $(length(seed)) elements and the model has $n_psi unknowns.")
        end
    end
    return 
end

function create_seeds(model::SDM{D, T}, n_seeds::Int, LB::T, UB::T) where {D, T}
    # Querying number of unknowns
    n_psi = dim_unknowns(model)

    seeds = Vector{Vector{T}}(undef, n_seeds)

    # Generate initial values in [-1e3, 1e3]
    for i = 1:n_seeds
        seeds[i] = rand(Distributions.Uniform(LB, UB), n_psi)
    end
    return seeds
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::AbstractOptimizationMethod{T},
                  optimizer::Optim.AbstractOptimizer, verbose::Int, i::Int) where T
                  
    return Optim.optimize(func, opt_method.seeds[i], optimizer,
                                                     Optim.Options(f_tol = opt_method.f_tol, 
                                                                   g_tol = opt_method.g_tol, 
                                                                   iterations = opt_method.iterations,
                                                                   show_trace = show_trace(verbose) ))
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::AbstractOptimizationMethod{T},
                  cons::Optim.TwiceDifferentiableConstraints,
                  optimizer::Optim.AbstractOptimizer, 
                  verbose::Int, i::Int) where T

    return Optim.optimize(func, cons, opt_method.seeds[i], optimizer,
                                                     Optim.Options(f_tol = opt_method.f_tol, 
                                                                   g_tol = opt_method.g_tol, 
                                                                   iterations = opt_method.iterations,
                                                                   show_trace = show_trace(verbose) ))
end

show_trace(verbose::Int) =  (verbose == 2 ? true : false)