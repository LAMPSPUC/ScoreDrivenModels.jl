RecipesBase.@recipe function f(b::ScoreDrivenModels.Backtest, name::String)
    xguide := "steps ahead"
    @series begin
        seriestype := :path
        label := "MAE " * name
        marker := :circle
        b.mae
    end
    @series begin
        seriestype := :path
        label := "Mean CRPS " * name
        marker := :circle
        b.mean_crps
    end
end