module ScoreDrivenModels

using Distributions, Optim, SpecialFunctions
using LinearAlgebra

import Base.length

const SCALINGS  = [0.0, 1/2, 1.0]
const PARAM_NUM_UB = 1e8
const ZERO_ROUNDING = 1e-8

# Core files
include("abstracts.jl")
include("utils.jl")
include("link_functions.jl")
include("score.jl")
include("MLE.jl")
include("opt_methods.jl")

# GAS
include("gas/gas.jl")
include("gas/initial_params.jl")
include("gas/simulate.jl")
include("gas/diagnostics.jl")
include("gas/univariate_score_driven_recursion.jl")

# Distributions
include("distributions/common_interface.jl")
include("distributions/normal.jl")
include("distributions/poisson.jl")
include("distributions/beta.jl")
include("distributions/lognormal.jl")
include("distributions/gamma.jl")
include("distributions/weibull.jl")

end