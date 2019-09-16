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
function RandomSeedsLBFGS(nseeds::Int, dim::Int; f_tol::Float64 = 1e-6, g_tol::Float64 = 1e-6, iterations::Int = 10^5)
    # sample seeds respecting spectral norm
    seeds = Vector{Vector{Float64}}(undef, nseeds)

    for i in 1:nseeds
        seeds[i] = rand(Uniform(0.0, 0.9), dim)
    end

    return RandomSeedsLBFGS(seeds; f_tol = f_tol, g_tol = g_tol, iterations = iterations)
end

function log_lik_sd_model(psitilde::Vector{T}, y::Vector{T}, sd_model::SDModel, 
                          initial_params::Vector{T}, unknowns_ω::Vector{Int},
                          unknowns_A::Vector{Int}, unknowns_B::Vector{Int}, n::Int) where T
    
    # Use the unkowns vectors to fill the right positions
    fill_psitilde!(sd_model, psitilde, unknowns_ω, unknowns_A, unknowns_B)

    if isnan(initial_params[1]) # Means default stationary initialization
        params = score_driven_recursion(sd_model, y)
    else
        params = score_driven_recursion(sd_model, y, initial_params)
    end

    return log_likelihood(sd_model.dist, y, params, n)
end

function estimate_SDModel!(sd_model::SDModel, y::Vector{T};
                           initial_params::Vector{Float64} = [NaN], # Means default stationary initialization
                           random_seeds_lbfgs::RandomSeedsLBFGS = RandomSeedsLBFGS(3, dimension_unkowns(sd_model)),
                           verbose::Int = 0) where T

    # Number of seed and number of params to estimate
    nseeds = length(random_seeds_lbfgs.seeds)
    len_ω = length(sd_model.ω)
    n = length(y)

    unknowns_ω = find_unknowns(sd_model.ω)
    unknowns_A = find_unknowns(sd_model.A)
    unknowns_B = find_unknowns(sd_model.B)
    len_unknowns = length(unknowns_ω) + length(unknowns_A) + length(unknowns_B)

    # Check if the model has no unknowns
    check_model_estimated(len_unknowns) && return sd_model

    # Guarantee that the seeds are in the right dimension
    @assert length(random_seeds_lbfgs.seeds[1]) == len_unknowns

    # optimize for each seed
    psi = Matrix{Float64}(undef, len_unknowns, nseeds)
    loglikelihood = Vector{Float64}(undef, nseeds)

    for i = 1:nseeds
        optseed = optimize(psi_tilde -> log_lik_sd_model(psi_tilde, y, sd_model, initial_params, unknowns_ω,
                                                         unknowns_A, unknowns_B, n), 
                                                         random_seeds_lbfgs.seeds[i],
                                                         LBFGS(), Optim.Options(f_tol = random_seeds_lbfgs.f_tol, 
                                                                                g_tol = random_seeds_lbfgs.g_tol, 
                                                                                iterations = random_seeds_lbfgs.iterations,
                                                                                show_trace = (verbose == 2 ? true : false) ))
        loglikelihood[i] = -optseed.minimum
        psi[:, i] = optseed.minimizer
        # print_loglikelihood(verbose, iseed, loglikelihood, t0)
    end

    bestpsi = psi[:, argmax(loglikelihood)]

    # return the estimated 
    fill_psitilde!(sd_model, bestpsi, unknowns_ω, unknowns_A, unknowns_B)

    println("Finished!")
end