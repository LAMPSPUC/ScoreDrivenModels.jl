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
        seeds[i] = rand(Uniform(-0.9, 0.9), dim)
    end

    return RandomSeedsLBFGS(seeds; f_tol = f_tol, g_tol = g_tol, iterations = iterations)
end

function log_lik_sd_model(psitilde::Vector{T}, y::Vector{T}, sd_model::SDModel, initial_params::Vector{T}, 
                          len_ω::Int, n::Int) where T
    # use len_ω to fill sd_model
    fill_ω!(sd_model, psitilde[1:len_ω])
    fill_A!(sd_model, psitilde[len_ω + 1:2len_ω])
    fill_B!(sd_model, psitilde[2len_ω + 1:3len_ω])

    if isnan(initial_params[1]) # Means default stationary initialization
        params = score_driven_recursion(sd_model, y)
    else
        params = score_driven_recursion(sd_model, y, initial_params)
    end

    return log_likelihood(sd_model.dist, y, params, n)
end

function estimate_SDModel!(sd_model::SDModel, y::Vector{T};
                           initial_params::Vector{Float64} = [NaN], # Means default stationary initialization
                           random_seeds_lbfgs::RandomSeedsLBFGS = RandomSeedsLBFGS(3, 3*length(sd_model.ω)),
                           verbose::Int == 0)

    # Number of seed and number of params to estimate
    nseeds = length(random_seeds_lbfgs.seeds)
    len_ω = length(sd_model.ω)
    n = length(y)

    # Guarantee that the seeds are in the right dimension
    @assert length(random_seeds_lbfgs.seeds[1]) == 3*len_ω

    # optimize for each seed
    psi = Matrix{Float64}(undef, 3*len_ω, nseeds)
    loglikelihood = Vector{Float64}(undef, nseeds)

    for i = 1:nseeds
        optseed = optimize(psitilde -> log_lik_sd_model(psitilde, y, sd_model, initial_params, len_ω, n), 
                            seeds[i],
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
    fill_ω!(sd_model, bestpsi[1:len_ω])
    fill_A!(sd_model, bestpsi[len_ω + 1:2len_ω])
    fill_B!(sd_model, bestpsi[2len_ω + 1:3len_ω])

    println("Finished!")
end