RecipesBase.@recipe function f(y::Vector, forec::Forecast)
    n = length(y)
    steps_ahead = length(forec.observation_forecast)
    forec_idx = collect(n + 1: n + steps_ahead)
    fillrange_color = :steelblue
    # Plot the series
    @series begin
        seriestype := :path
        seriescolor := "black"
        return y
    end
    # Plot the forecast
    @series begin
        seriescolor := "blue"
        return forec_idx, forec.observation_forecast
    end
    # Plot the prediction interval
    q975 = Vector{Float64}(undef, steps_ahead)
    q025 = Vector{Float64}(undef, steps_ahead)
    q900 = Vector{Float64}(undef, steps_ahead)
    q100 = Vector{Float64}(undef, steps_ahead)
    for i in 1:steps_ahead
        q975[i] = quantile(forec.observation_scenarios[i, :], 0.975)
        q025[i] = quantile(forec.observation_scenarios[i, :], 0.025)
        q900[i] = quantile(forec.observation_scenarios[i, :], 0.9)
        q100[i] = quantile(forec.observation_scenarios[i, :], 0.1)
    end
    @series begin
        seriestype := :path
        seriescolor := fillrange_color
        fillcolor := fillrange_color
        fillalpha := 0.5
        fillrange := q975
        label := ""
        return forec_idx, q025
    end
    @series begin
        seriestype := :path
        seriescolor := fillrange_color
        fillcolor := fillrange_color
        fillalpha := 0.5
        fillrange := q900
        label := ""
        return forec_idx, q100
    end
end
