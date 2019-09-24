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

function log_lik_gas_sarima(psitilde::Vector{T}, y::Vector{T}, gas_sarima::GAS_Sarima, 
                            initial_params::Vector{Vector{Float64}}, unknowns_gas_sarima::Unknowns_GAS_Sarima, n::Int) where T
    
    # Use the unkowns vectors to fill the right positions
    fill_psitilde!(gas_sarima, psitilde, unknowns_gas_sarima)

    if isnan(initial_params[1][1]) # Means default stationary initialization
        params = score_driven_recursion(gas_sarima, y)
    else
        params = score_driven_recursion(gas_sarima, y, initial_params)
    end

    return log_likelihood(gas_sarima.dist, y, params, n)
end

function estimate!(gas_sarima::GAS_Sarima, y::Vector{T};
                   initial_params::Vector{Vector{Float64}} = [[NaN]], # Means default initializations
                   random_seeds_lbfgs::RandomSeedsLBFGS = RandomSeedsLBFGS(3, dimension_unkowns(gas_sarima)),
                   verbose::Int = 0) where T

    # Number of seed and number of params to estimate
    nseeds = length(random_seeds_lbfgs.seeds)
    n = length(y)

    unknowns_gas_sarima = find_unknowns(gas_sarima)
    len_unknowns = length(unknowns_gas_sarima)

    # Check if the model has no unknowns
    check_model_estimated(len_unknowns) && return gas_sarima

    # Guarantee that the seeds are in the right dimension
    @assert length(random_seeds_lbfgs.seeds[1]) == len_unknowns

    # optimize for each seed
    psi = Vector{Vector{Float64}}(undef, 0)
    loglikelihood = Vector{Float64}(undef, 0)

    for i = 1:nseeds
        try 
            optseed = optimize(psi_tilde -> log_lik_gas_sarima(psi_tilde, y, gas_sarima, initial_params, unknowns_gas_sarima, n), 
                                                               random_seeds_lbfgs.seeds[i],
                                                               LBFGS(), Optim.Options(f_tol = random_seeds_lbfgs.f_tol, 
                                                                                      g_tol = random_seeds_lbfgs.g_tol, 
                                                                                      iterations = random_seeds_lbfgs.iterations,
                                                                                      show_trace = (verbose == 2 ? true : false) ))
            push!(loglikelihood, -optseed.minimum)
            push!(psi, optseed.minimizer)
            # print_loglikelihood(verbose, iseed, loglikelihood, t0)
            println("seed $i of $nseeds - $(-optseed.minimum)")
        catch
            println("seed $i diverged")
        end
    end

    bestpsi = psi[argmax(loglikelihood)]

    # return the estimated 
    fill_psitilde!(gas_sarima, bestpsi, unknowns_gas_sarima)

    println("Finished!")
end