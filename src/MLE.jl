export estimate!

"""
    AbstractOptimizationMethod
Abstract type used to implement an interface for generic optimization methods.
"""
abstract type AbstractOptimizationMethod end

"""
TODO
"""
mutable struct RandomSeedsLBFGS <: AbstractOptimizationMethod
    f_tol::Float64
    g_tol::Float64
    iterations::Int
    seeds::Vector{Vector{Float64}}

    function RandomSeedsLBFGS(seeds::Vector{Vector{Float64}}; f_tol::Float64 = 1e-6, g_tol::Float64 = 1e-6, iterations::Int = 10^5)
        return new(f_tol, g_tol, 1e5, seeds)
    end
end
function RandomSeedsLBFGS(nseeds::Int, dim::Int; f_tol::Float64 = 1e-6, g_tol::Float64 = 1e-6, iterations::Int = 10^5,
                          LB::Float64 = 0.0, UB::Float64 = 0.6)
    seeds = Vector{Vector{Float64}}(undef, nseeds)

    for i in 1:nseeds
        seeds[i] = rand(Uniform(LB, UB), dim)
    end

    return RandomSeedsLBFGS(seeds; f_tol = f_tol, g_tol = g_tol, iterations = iterations)
end

function log_lik(psitilde::Vector{T}, y::Vector{T}, gas::GAS{D, T}, 
                 initial_params::Vector{Vector{Float64}}, unknowns::Unknowns_GAS, n::Int) where {D, T}
    
    # Use the unkowns vectors to fill the right positions
    fill_psitilde!(gas, psitilde, unknowns)

    if isnan(initial_params[1][1]) # Means default stationary initialization
        params = score_driven_recursion(gas, y)
    else
        params = score_driven_recursion(gas, y, initial_params)
    end

    return log_likelihood(D, y, params, n)
end

function estimate!(gas::GAS, y::Vector{T};
                   initial_params::Vector{Vector{Float64}} = [[NaN]], # Means default initializations
                   random_seeds_lbfgs::RandomSeedsLBFGS = RandomSeedsLBFGS(3, dim_unknowns(gas)),
                   verbose::Int = 0) where T

    # Number of seed and number of params to estimate
    nseeds = length(random_seeds_lbfgs.seeds)
    n = length(y)

    unknowns = find_unknowns(gas)
    len_unknowns = length(unknowns)

    # Check if the model has no unknowns
    check_model_estimated(len_unknowns) && return gas

    # Guarantee that the seeds are in the right dimension
    @assert length(random_seeds_lbfgs.seeds[1]) == len_unknowns

    # optimize for each seed
    psi = Vector{Vector{Float64}}(undef, 0)
    loglikelihood = Vector{Float64}(undef, 0)
    optseeds = Vector{Optim.OptimizationResults}(undef, 0)

    for i = 1:nseeds
        try 
            optseed = optimize(psi_tilde -> log_lik(psi_tilde, y, gas, initial_params, unknowns, n), 
                                                    random_seeds_lbfgs.seeds[i],
                                                    LBFGS(), Optim.Options(f_tol = random_seeds_lbfgs.f_tol, 
                                                                           g_tol = random_seeds_lbfgs.g_tol, 
                                                                           iterations = random_seeds_lbfgs.iterations,
                                                                           show_trace = (verbose == 2 ? true : false) ))
            push!(loglikelihood, -optseed.minimum)
            push!(psi, optseed.minimizer)
            push!(optseeds, optseed)
            println("seed $i of $nseeds - $(-optseed.minimum)")
        catch
            println("seed $i diverged")
        end
    end

    if isempty(loglikelihood) 
        println("No seed converged.")
        return
    end

    best_llk = argmax(loglikelihood)
    bestpsi = psi[best_llk]
    if verbose >= 1
        println("\nBest seed optimization result:")
        println(optseeds[best_llk])
    end

    # return the estimated 
    fill_psitilde!(gas, bestpsi, unknowns)

    println("Finished!")
end