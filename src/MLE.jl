export estimate!

const DEFAULT_INITIAL_PARAM = NaN.*ones(1, 1)
const DEFAULT_NUM_SEEDS = 3

struct EstimationResults{T <: AbstractFloat}
    aic::T
    bic::T
    llk::T
    minimizer::Vector{T}
    numerical_hessian::Matrix{T}
end

function log_lik(psitilde::Vector{T}, y::Vector{T}, sdm::SDM{D, T}, 
                 initial_params::Vector{Vector{T}}, unknowns::Unknowns_SDM, n::Int) where {D, T}
    return error("log_lik not defined for a model of type ", typeof(sdm))
end

function AIC(n_unknowns::Int, log_lik::T) where T
    return T(2 * n_unknowns - 2 * log_lik)
end

function BIC(n::Int, n_unknowns::Int, log_lik::T) where T
    return T(log(n) * n_unknowns - 2 * log_lik)
end

function estimate!(sdm::SDM{D, T}, y::Vector{T};
                   initial_params::Matrix{T} = DEFAULT_INITIAL_PARAM,
                   opt_method::AbstractOptimizationMethod = LBFGS(sdm, DEFAULT_NUM_SEEDS),
                   verbose::Int = 0) where {D, T}

    # Number of seed and number of params to estimate
    nseeds = length(opt_method.seeds)
    n = length(y)

    unknowns = find_unknowns(sdm)
    n_unknowns = length(unknowns)

    # Check if the model has no unknowns
    check_model_estimated(n_unknowns) && return sdm

    # optimize for each seed
    psi = Vector{Vector{T}}(undef, 0)
    numerical_hessian = Vector{Matrix{T}}(undef, 0)
    loglikelihood = Vector{T}(undef, 0)
    optseeds = Vector{Optim.OptimizationResults}(undef, 0)

    for i = 1:nseeds
        try 
            func = TwiceDifferentiable(psi_tilde -> log_lik(psi_tilde, y, sdm, initial_params, unknowns, n), opt_method.seeds[i])
            optseed = optimize(func, opt_method.seeds[i],
                                     opt_method.method, Optim.Options(f_tol = opt_method.f_tol, 
                                                                      g_tol = opt_method.g_tol, 
                                                                      iterations = opt_method.iterations,
                                                                      show_trace = (verbose == 2 ? true : false) ))
            push!(loglikelihood, -optseed.minimum)
            push!(psi, optseed.minimizer)
            push!(numerical_hessian, Optim.hessian!(func, optseed.minimizer))
            push!(optseeds, optseed)
            println("seed $i of $nseeds - $(-optseed.minimum)")
        catch err
            println(err)
            println("seed $i diverged")
        end
    end

    if isempty(loglikelihood) 
        println("No seed converged.")
        return
    end

    best_llk, best_seed = findmax(loglikelihood)
    num_hessian = numerical_hessian[best_seed]
    best_psi = psi[best_seed]
    aic = AIC(n_unknowns, best_llk)
    bic = BIC(n, n_unknowns, best_llk)

    if verbose >= 1
        println("\nBest seed optimization result:")
        println(optseeds[best_seed])
    end

    # return the estimated 
    fill_psitilde!(sdm, best_psi, unknowns)

    println("Finished!")
    return EstimationResults{T}(aic, bic, best_llk, best_psi, num_hessian)
end