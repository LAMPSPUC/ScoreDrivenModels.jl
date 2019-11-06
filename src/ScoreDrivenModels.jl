module ScoreDrivenModels

using Distributions, Optim, SpecialFunctions
using LinearAlgebra

import Base.length

# Core files
include("abstracts.jl")
include("GAS.jl")
include("univariate_score_driven_recursion.jl")
include("utils.jl")
include("sample.jl")
include("link_functions.jl")
include("score.jl")
include("initial_params.jl")
include("simulate.jl")
include("MLE.jl")
include("opt_methods.jl")
include("diagnostics.jl")

# Distributions
include("distributions/common_interface.jl")
include("distributions/normal.jl")
include("distributions/poisson.jl")
include("distributions/beta.jl")
include("distributions/log_normal.jl")
include("distributions/gamma.jl")

end