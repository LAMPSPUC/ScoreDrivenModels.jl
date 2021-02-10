module ScoreDrivenModels

using Distributions, Optim, SpecialFunctions
using LinearAlgebra, Printf

import Base: length, deepcopy, show

const SCALINGS  = [0.0, 1/2, 1.0]
const BIG_NUM = 1e8
const SMALL_NUM = 1e-8

# Core files
include("abstracts.jl")
include("utils.jl")
include("link_functions.jl")
include("score.jl")
include("model.jl")
include("initial_params.jl")
include("simulate.jl")
include("diagnostics.jl")
include("univariate_score_driven_recursion.jl")
include("MLE.jl")
include("backtest.jl")
include("prints.jl")

# Optimization methods
include("opt_methods/common_methods.jl")
include("opt_methods/LBFGS.jl")
include("opt_methods/IPNewton.jl")
include("opt_methods/NelderMead.jl")

# Distributions
include("distributions/non_native_dists.jl")
include("distributions/common_interface.jl")
include("distributions/beta.jl")
include("distributions/betafour.jl")
include("distributions/exponential.jl")
include("distributions/gamma.jl")
include("distributions/logitnormal.jl")
include("distributions/lognormal.jl")
include("distributions/negativebinomial.jl")
include("distributions/normal.jl")
include("distributions/poisson.jl")
include("distributions/tdist.jl")
include("distributions/tdistlocationscale.jl")
include("distributions/weibull.jl")

end
