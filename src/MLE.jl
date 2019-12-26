export fit, fit!, fit_stats

const DEFAULT_INITIAL_PARAM = NaN.*ones(1, 1)
const DEFAULT_NUM_SEEDS = 3
const DEFAULT_VERBOSE = 0
const VARIANCE_ZERO = 1e-10

struct FittedSDM{D <: Distribution, T <: AbstractFloat}
    num_obs::Integer
    unknowns::UnknownsSDM
    aic::T
    bic::T
    llk::T
    coefs::Vector{T}
    numerical_hessian::Matrix{T}
end

struct CoefsStatsSDM{T <: AbstractFloat}
    unknowns::UnknownsSDM
    coefs::Vector{T}
    std_errors::Vector{T}
    t_stat::Vector{T}
    p_values::Vector{T}
end

struct EstimationStatsSDM{D <: Distribution, T <: AbstractFloat}
    num_obs::Integer
    loglikelihood::T
    aic::T
    bic::T
    np::T
    coefs_stats::CoefsStatsSDM{T}
end

function fit_stats(f::FittedSDM{D, T}) where {D, T}
    estim_results = eval_coefs_stats(f)
    np = length(f.unknowns)
    return EstimationStatsSDM{D, T}(f.num_obs, f.llk, f.aic, f.bic, np, estim_results)
end

function eval_coefs_stats(f::FittedSDM{D, T}) where {D, T}
    np = length(f.unknowns)
    inv_H = inv(f.numerical_hessian)
    vars = diag(inv_H)
    for i in eachindex(vars)
        if vars[i] <= VARIANCE_ZERO 
            vars[i] = VARIANCE_ZERO
        end
    end
    std_errors = sqrt.(vars)
    t_stats = f.coefs ./ std_errors

    # Calculate p-values of the the t statistics
    t_dist = TDist(np)
    p_values = 1 .- 2*abs.(cdf.(t_dist, t_stats) .- 0.5)

    return CoefsStatsSDM{T}(f.unknowns, f.coefs, std_errors, t_stats, p_values)
end

mutable struct AuxEstimation{T <: AbstractFloat}
    psi::Vector{Vector{T}}
    numerical_hessian::Vector{Matrix{T}}
    loglikelihood::Vector{T}
    opt_result::Vector{Optim.OptimizationResults}

    function AuxEstimation{T}() where T
        return new(
            Vector{Vector{T}}(undef, 0), #psi
            Vector{Matrix{T}}(undef, 0), 
            Vector{T}(undef, 0), # loglikelihood
            Vector{Optim.OptimizationResults}(undef, 0) # opt_result
            )
    end
end

function AIC(n_unknowns::Int, log_lik::T) where T
    return T(2 * n_unknowns - 2 * log_lik)
end

function BIC(n::Int, n_unknowns::Int, log_lik::T) where T
    return T(log(n) * n_unknowns - 2 * log_lik)
end

function update_aux_estimation!(aux_est::AuxEstimation{T}, func::Optim.TwiceDifferentiable,
                                opt_result::Optim.OptimizationResults) where T
                                
    push!(aux_est.loglikelihood, -opt_result.minimum)
    push!(aux_est.psi, opt_result.minimizer)
    push!(aux_est.numerical_hessian, Optim.hessian!(func, opt_result.minimizer))
    push!(aux_est.opt_result, opt_result)
    return
end

function fit(sdm::SDM{D, T}, y::Vector{T};
             initial_params::Matrix{T} = DEFAULT_INITIAL_PARAM,
             opt_method::AbstractOptimizationMethod = NelderMead(sdm, DEFAULT_NUM_SEEDS),
             verbose::Int = DEFAULT_VERBOSE) where {D, T}

    # Number of seed and number of params to estimate
    n_seeds = length(opt_method.seeds)
    n = length(y)
    
    unknowns = find_unknowns(sdm)
    n_unknowns = length(unknowns)
    
    # Check if the model has no unknowns
    check_model_estimated(n_unknowns) && return sdm

    # Create a copy of the model to estimate
    sdm_fit = deepcopy(sdm)
    
    # optimize for each seed
    aux_est = AuxEstimation{T}()

    for i = 1:n_seeds
        try 
            func = TwiceDifferentiable(psi_tilde -> log_lik(psi_tilde, y, sdm_fit, initial_params, unknowns, n), opt_method.seeds[i])
            opt_result = optimize(func, opt_method, verbose, i)
            update_aux_estimation!(aux_est, func, opt_result)
            println("seed $i of $n_seeds - $(-opt_result.minimum)")
        catch err
            println(err)
            println("seed $i diverged")
        end
    end

    if isempty(aux_est.loglikelihood) 
        println("No seed converged.")
        return
    end

    best_llk, best_seed = findmax(aux_est.loglikelihood)
    num_hessian = aux_est.numerical_hessian[best_seed]
    coefs = aux_est.psi[best_seed]
    aic = AIC(n_unknowns, best_llk)
    bic = BIC(n, n_unknowns, best_llk)

    if verbose >= 1
        println("\nBest seed optimization result:")
        println(aux_est.opt_result[best_seed])
    end

    println("Finished!")
    return FittedSDM{D, T}(n, unknowns, aic, bic, best_llk, coefs, num_hessian)
end

function fit!(sdm::SDM{D, T}, y::Vector{T};
              initial_params::Matrix{T} = DEFAULT_INITIAL_PARAM,
              opt_method::AbstractOptimizationMethod = LBFGS(sdm, DEFAULT_NUM_SEEDS),
              verbose::Int = DEFAULT_VERBOSE) where {D, T}

    unknowns = find_unknowns(sdm)
    # Check if the model has no unknowns
    n_unknowns = length(unknowns)
    check_model_estimated(n_unknowns) && return sdm
    
    f = fit(sdm, y; initial_params = initial_params, opt_method = opt_method, verbose = verbose)
    fill_psitilde!(sdm, f.coefs, unknowns)
    return f
end