RecipesBase.@recipe function f(fi::Fitted)
    layout := (2, 2)
    acf = autocor(fi.pearson_residuals)[2:end]
    @series begin
        seriestype := :path
        label := ""
        seriescolor := "black"
        subplot := 1
        marker := :circle
        fi.pearson_residuals
    end
    @series begin
        seriestype := :bar
        label := ""
        seriescolor := "black"
        subplot := 2
        acf
    end
    ub = ones(length(acf)) * 1.96 / sqrt(fi.num_obs)
    lb = ones(length(acf)) * -1.96 / sqrt(fi.num_obs)
    @series begin
        seriestype := :path
        linestyle := :dash
        seriescolor := "red"
        label := ""
        subplot := 2
        ub
    end
    @series begin
        seriestype := :path
        linestyle := :dash
        label := ""
        seriescolor := "red"
        subplot := 2
        lb
    end
    @series begin
        seriestype := :histogram
        label := ""
        seriescolor := "black"
        subplot := 3
        fi.pearson_residuals
    end
    qqpair = qqbuild(Normal(), fi.pearson_residuals)
    @series begin
        seriestype := :scatter
        label := ""
        seriescolor := "black"
        subplot := 4
        qqpair.qx, qqpair.qy
    end
    @series begin
        seriestype := :path
        seriescolor := "red"
        label := ""
        subplot := 4
        collect(-3:0.01:3), collect(-3:0.01:3)
    end
end