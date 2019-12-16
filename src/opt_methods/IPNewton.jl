export IPNewton

mutable struct IPNewton{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    ub::Vector{T}
    lb::Vector{T}
    iterations::Int
    n_seeds::Int
    seeds::Vector{Vector{T}}
    method::Optim.AbstractOptimizer
end

"""
    IPNewton(model::SDM, args...; kwargs...)

If an `Int` is provided the method will sample random seeds and use them as initial points for Optim IPNewton method.
If a `Vector{Vector{T}}` is provided it will use them as initial points for Optim IPNewton method. Notice that the 
IPNewton method supports upper bounds and lower bound for the optimization.
"""
function IPNewton(model::SDM{D, T}, n_seeds::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6),
                                            ub::Vector{T} = Inf.*ones(dim_unknowns(model)),  
                                            lb::Vector{T} = -Inf.*ones(dim_unknowns(model)),
                                            iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6) where {D, T}

    # Querying number of unknowns
    n_psi = dim_unknowns(model)
    @assert length(ub) == length(lb) == n_psi

    seeds = Vector{Vector{T}}(undef, n_seeds)

    # Generate initial values in [-1e3, 1e3]
    for i = 1:n_seeds
        seeds[i] = rand(Distributions.Uniform(LB, UB), n_psi)
    end
    
    return IPNewton{T}(f_tol, g_tol, ub, lb, iterations, n_seeds, seeds, Optim.IPNewton())
end

function IPNewton(model::SDM{D, T}, seeds::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                            ub::Vector{T} = Inf.*ones(dim_unknowns(model)),  
                                            lb::Vector{T} = -Inf.*ones(dim_unknowns(model)),
                                            iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6) where {D, T}

    # Querying number of unknowns
    n_psi = dim_unknowns(model)
    @assert length(ub) == length(lb) == n_psi

    n_seeds = length(seeds)

    for (i, seed) in enumerate(seeds)
        if length(seed) != n_psi
            error("Seed $i has $(length(seed)) elements and the model has $n_psi unknowns.")
        end
    end
    
    return IPNewton{T}(f_tol, g_tol, ub, lb, iterations, n_seeds, seeds, Optim.IPNewton())
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::IPNewton{T}, verbose::Int, i::Int) where T
    cons = TwiceDifferentiableConstraints(opt_method.lb, opt_method.ub)
    return Optim.optimize(func, cons, opt_method.seeds[i], opt_method.method,
                                                     Optim.Options(f_tol = opt_method.f_tol, 
                                                                   g_tol = opt_method.g_tol, 
                                                                   iterations = opt_method.iterations,
                                                                   show_trace = (verbose == 2 ? true : false) ))
end