# Define TDistLocationScale as the location scale transformation of the Distributions.jl TDist
export TDistLocationScale
TDistLocationScale = AffineDistribution{Float64,value_support(TDist),TDist{Float64}}

export BetaFourParameters
BetaFourParameters = AffineDistribution{Float64,value_support(Beta),Beta{Float64}}
