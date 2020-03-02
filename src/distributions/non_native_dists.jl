# Define TDistLocationScale as the location scale transformation of the Distributions.jl TDist
export TDistLocationScale
TDistLocationScale = LocationScale{Float64,TDist{Float64}}

# Define BetaLocationScale as the location scale transformation of the Distributions.jl TDist
export BetaLocationScale
BetaLocationScale = LocationScale{Float64,Beta{Float64}}