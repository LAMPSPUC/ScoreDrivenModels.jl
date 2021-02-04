# Define TDistLocationScale as the location scale transformation of the Distributions.jl TDist
export TDistLocationScale
TDistLocationScale = LocationScale{Float64,TDist{Float64}}

export BetaFourParameters
BetaFourParameters = LocationScale{Float64,Beta{Float64}}