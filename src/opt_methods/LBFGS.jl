export LBFGS

mutable struct LBFGS{T <: Real} <: AbstractOptimizationMethod{T}
    f_tol::T
    g_tol::T
    iterations::Int
    n_seeds::Int
    seeds::Vector{Vector{T}}
    method::Optim.AbstractOptimizer
end

"""
    LBFGS(model::SDM, args...; kwargs...)

If an `Int` is provided the method will sample random seeds and use them as initial points for Optim LBFGS method.
If a `Vector{Vector{T}}` is provided it will use them as initial points for Optim LBFGS method.
"""
function LBFGS(model::SDM{D, T}, n_seeds::Int; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                         iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6) where {D, T}

    # Querying number of unknowns
    n_psi = dim_unknowns(model)

    seeds = Vector{Vector{T}}(undef, n_seeds)

    # Generate initial values in [-1e3, 1e3]
    for i = 1:n_seeds
        seeds[i] = rand(Distributions.Uniform(LB, UB), n_psi)
    end
    
    return LBFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds, Optim.LBFGS())
end

function LBFGS(model::SDM{D, T}, seeds::Vector{Vector{T}}; f_tol::T = T(1e-6), g_tol::T = T(1e-6), 
                                        iterations::Int = 10^5, LB::T = 0.0, UB::T = 0.6) where {D, T}

    # Querying number of unknowns
    n_psi = dim_unknowns(model)

    n_seeds = length(seeds)

    for (i, seed) in enumerate(seeds)
        if length(seed) != n_psi
            error("Seed $i has $(length(seed)) elements and the model has $n_psi unknowns.")
        end
    end
    
    return LBFGS{T}(f_tol, g_tol, iterations, n_seeds, seeds, Optim.LBFGS())
end

function optimize(func::Optim.TwiceDifferentiable, opt_method::LBFGS{T}, verbose::Int, i::Int) where T
    return Optim.optimize(func, opt_method.seeds[i], opt_method.method,
                                                     Optim.Options(f_tol = opt_method.f_tol, 
                                                                   g_tol = opt_method.g_tol, 
                                                                   iterations = opt_method.iterations,
                                                                   show_trace = (verbose == 2 ? true : false) ))
end