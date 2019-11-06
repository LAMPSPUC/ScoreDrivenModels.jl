module ScoreDrivenModels

using Distributions, Optim, SpecialFunctions
using LinearAlgebra

import Base.length

# Core files
include("abstracts.jl")
include("utils.jl")
include("sample.jl")
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
include("distributions/log_normal.jl")
include("distributions/gamma.jl")

end