export estimate!

function log_lik(psitilde::Vector{T}, y::Vector{T}, sdm::SDM{D, T}, 
                 initial_params::Vector{Vector{T}}, unknowns::Unknowns_SDM, n::Int) where {D, T}
    return error("log_lik not defined for a model of type ", typeof(sdm))
end

function estimate!(sdm::SDM{D, T}, y::Vector{T};
                   initial_params::Vector{Vector{Float64}} = [[NaN]], # Means default initializations
                   opt_method::AbstractOptimizationMethod = LBFGS(sdm, 3),
                   verbose::Int = 0) where {D, T}

    # Number of seed and number of params to estimate
    nseeds = length(opt_method.seeds)
    n = length(y)

    unknowns = find_unknowns(sdm)
    len_unknowns = length(unknowns)

    # Check if the model has no unknowns
    check_model_estimated(len_unknowns) && return sdm

    # Guarantee that the seeds are in the right dimension
    @assert length(opt_method.seeds[1]) == len_unknowns

    # optimize for each seed
    psi = Vector{Vector{Float64}}(undef, 0)
    loglikelihood = Vector{Float64}(undef, 0)
    optseeds = Vector{Optim.OptimizationResults}(undef, 0)

    for i = 1:nseeds
        try 
            optseed = optimize(psi_tilde -> log_lik(psi_tilde, y, sdm, initial_params, unknowns, n), 
                                                    opt_method.seeds[i],
                                                    opt_method.method, Optim.Options(f_tol = opt_method.f_tol, 
                                                                                     g_tol = opt_method.g_tol, 
                                                                                     iterations = opt_method.iterations,
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
    fill_psitilde!(sdm, bestpsi, unknowns)

    println("Finished!")
end