# Define Std as the location scale transformation of the Distributions.jl TDist
export LocationScaleTDist
LocationScaleTDist = LocationScale{Float64,TDist{Float64}}