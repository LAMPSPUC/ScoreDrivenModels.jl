export fit, fit!, results

const DEFAULT_INITIAL_PARAM = NaN.*ones(1, 1)
const DEFAULT_NUM_SEEDS = 3
const DEFAULT_VERBOSE = 1
const VARIANCE_ZERO = 1e-10

struct Fitted{D <: Distribution, T <: AbstractFloat}
    num_obs::Integer
    unknowns::Unknowns
    aic::T
    bic::T
    llk::T
    coefs::Vector{T}
    numerical_hessian::Matrix{T}
    quantile_residuals::Vector{T}
end

struct CoefsStats{T <: AbstractFloat}
    unknowns::Unknowns
    coefs::Vector{T}
    std_errors::Vector{T}
    t_stat::Vector{T}
    p_values::Vector{T}
end

struct EstimationStats{D <: Distribution, T <: AbstractFloat}
    num_obs::Integer
    loglikelihood::T
    aic::T
    bic::T
    np::T
    jarquebera_p_value::T
    coefs_stats::CoefsStats{T}
end

function results(f::Fitted{D, T}) where {D, T}
    estim_results = eval_coefs_stats(f)
    np = length(f.unknowns)
    jarquebera_p_value = pvalue(JarqueBeraTest(f.quantile_residuals))
    return EstimationStats{D, T}(f.num_obs, f.llk, f.aic, f.bic, np,
                                 jarquebera_p_value, estim_results)
end

function eval_coefs_stats(f::Fitted{D, T}) where {D, T}
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

    return CoefsStats{T}(f.unknowns, f.coefs, std_errors, t_stats, p_values)
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

    push!(aux_est.numerical_hessian, Optim.hessian!(func, opt_result.minimizer))
    push!(aux_est.psi, opt_result.minimizer)
    push!(aux_est.loglikelihood, -opt_result.minimum)
    push!(aux_est.opt_result, opt_result)
    return
end

function fit!(gas::ScoreDrivenModel{D, T}, y::Vector{T};
             initial_params::Matrix{T} = DEFAULT_INITIAL_PARAM,
             opt_method::AbstractOptimizationMethod = NelderMead(gas, DEFAULT_NUM_SEEDS),
             verbose::Int = DEFAULT_VERBOSE,
             throw_errors::Bool = false,
             time_limit_sec::Int = 10^8) where {D, T}

    unknowns = find_unknowns(gas)
    # Check if the model has no unknowns
    n_unknowns = length(unknowns)
    check_model_estimated(n_unknowns) && return gas

    verbose in [0, 1, 2, 3] || throw(ErrorException, "verbose argument must be in [0, 1, 2, 3]")
    # Number of initial_points and number of params to estimate
    n_initial_points = length(opt_method.initial_points)
    n = length(y)

    unknowns = find_unknowns(gas)
    n_unknowns = length(unknowns)

    # Check if the model has no unknowns
    check_model_estimated(n_unknowns) && return gas

    # Create a copy of the model to estimate
    gas_fit = deepcopy(gas)

    # optimize for each initial_point
    aux_est = AuxEstimation{T}()

    for i = 1:n_initial_points
        try
            func = TwiceDifferentiable(psi_tilde -> log_lik(psi_tilde, y, gas_fit,
                                                        initial_params, unknowns, n),
                                                        opt_method.initial_points[i])
            opt_result = optimize(func, opt_method, verbose, i, time_limit_sec)
            update_aux_estimation!(aux_est, func, opt_result)
            verbose >= 1 && println("Round $i of $n_initial_points - Log-likelihood: $(-opt_result.minimum * length(y))")
        catch err
            println(err)
            if throw_errors
                throw(err)
            end
            verbose >= 1 && println("Round $i diverged")
        end
    end

    if isempty(aux_est.loglikelihood)
        verbose >= 1 && println("No initial point converged.")
        return
    end

    best_llk, best_seed = findmax(aux_est.loglikelihood)
    # We optimize the average loglikelihood inside log_lik
    best_llk = best_llk * length(y)
    num_hessian = aux_est.numerical_hessian[best_seed]
    coefs = aux_est.psi[best_seed]
    aic = AIC(n_unknowns, best_llk)
    bic = BIC(n, n_unknowns, best_llk)

    if verbose >= 2
        println("\nBest optimization result:")
        println(aux_est.opt_result[best_seed])
    end

    fill_psitilde!(gas, coefs, unknowns)

    # Calculate pearson residuals
    pearson_res = isnan(initial_params[1]) ?
                quantile_residuals(y, gas) :
                quantile_residuals(y, gas; initial_params = initial_params)

    return Fitted{D, T}(n, unknowns, aic, bic, best_llk, coefs, num_hessian, pearson_res)
end
