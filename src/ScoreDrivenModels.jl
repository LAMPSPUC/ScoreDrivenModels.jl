module ScoreDrivenModels

using Distributions, Optim, SpecialFunctions
using LinearAlgebra

import Base.length

# Core files
include("structures.jl")
include("univariate_score_driven_model.jl")
include("utils.jl")
include("sample.jl")
include("link_functions.jl")
include("score.jl")
include("initial_params.jl")
include("simulate.jl")
include("MLE.jl")

# Distributions
include("distributions/normal.jl")
include("distributions/poisson.jl")
include("distributions/beta.jl")

end