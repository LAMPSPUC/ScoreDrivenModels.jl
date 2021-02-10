export backtest

struct Backtest
    abs_errors::Matrix{Float64}
    crps_scores::Matrix{Float64}
    function Backtest(n::Int, steps_ahead::Int)
        abs_errors = Matrix{Float64}(undef, n, steps_ahead)
        crps_scores = Matrix{Float64}(undef, n, steps_ahead)
        return new(abs_errors, crps_scores)
    end
end

discrete_crps_indicator_function(val::T, z::T) where {T} = val < z
function crps(val::T, scenarios::Vector{T}) where {T}
    sorted_scenarios = sort(scenarios)
    m = length(scenarios)
    crps_score = zero(T)
    for i = 1:m
        crps_score +=
            (sorted_scenarios[i] - val) *
            (m * discrete_crps_indicator_function(val, sorted_scenarios[i]) - i + 0.5)
    end
    return (2 / m^2) * crps_score
end
evaluate_abs_error(y::Vector{T}, forecast::Vector{T}) where T = abs.(y - forecast)
function evaluate_crps(y::Vector{T}, scenarios::Matrix{T}) where {T}
    crps_scores = Vector{T}(undef, length(y))
    for k = 1:length(y)
        crps_scores[k] = crps(y[k], scenarios[k, :])
    end
    return crps_scores
end

"""
TODO
"""
function backtest(gas::Model{<:Distribution, T}, y::Vector{T}, steps_ahead::Int, start_idx::Int;
                  S::Int = 10_000,
                  initial_params = stationary_initial_params(gas),
                  opt_method = NelderMead(gas, 3)) where T
    num_mle = length(y) - start_idx - steps_ahead
    b = Backtest(num_mle, steps_ahead)
    for i in 1:num_mle
        println("Backtest: step $i of $num_mle")
        gas_to_fit = deepcopy(gas)
        y_to_fit = y[1:start_idx - 1 + i]
        y_to_verify = y[start_idx + i:start_idx - 1 + i + steps_ahead]
        ScoreDrivenModels.fit!(gas_to_fit, y_to_fit; initial_params=initial_params, opt_method=opt_method)
        forec = forecast(y_to_fit, gas_to_fit, steps_ahead; S=S, initial_params=initial_params)
        abs_errors = evaluate_abs_error(y_to_verify, forec.observation_forecast)
        crps_scores = evaluate_crps(y_to_verify, forec.observation_scenarios)
        b.abs_errors[i, :] = abs_errors
        b.crps_scores[i, :] = crps_scores
    end
    return b
end