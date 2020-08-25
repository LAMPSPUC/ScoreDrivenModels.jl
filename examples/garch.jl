using DelimitedFiles, ScoreDrivenModels, Statistics

# This is a more advanced example. We want to specify a GARCH model using the GAS framework.
# To that end, we need to overwrite the default link-unlink interface as it would lead to
# a different parametrization.
function ScoreDrivenModels.link!(param_tilde::Matrix{T}, ::Type{Normal}, param::Matrix{T}, t::Int) where T
    param_tilde[t, 1] = link(IdentityLink, param[t, 1])
    param_tilde[t, 2] = link(IdentityLink, param[t, 2])
    return
end
function ScoreDrivenModels.unlink!(param::Matrix{T}, ::Type{Normal}, param_tilde::Matrix{T}, t::Int) where T
    param[t, 1] = unlink(IdentityLink, param_tilde[t, 1])
    param[t, 2] = unlink(IdentityLink, param_tilde[t, 2])
    return
end
function ScoreDrivenModels.jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Normal}, param::Matrix{T}, t::Int) where T
    aux.jac[1] = jacobian_link(IdentityLink, param[t, 1])
    aux.jac[2] = jacobian_link(IdentityLink, param[t, 2])
    return
end

# Load (TODO: what is this data?) data
y = vec(readdlm("../test/data/BG96.csv"))

# Set initial parameters as observed mean and variance of the series
initial_params = [mean(y) var(y)]

# We can set upper and lower bounds for the parameters
ub = [1.0, 1.0, 0.5, 1.0]
lb = [-1.0, 0.0, 0.0, 0.5]

# Specify GAS model: a normal model with time-varying σ and lag 1
gas = Model(1, 1, Normal, 1.0, time_varying_params = [2])

# Set initial point for the optimization
initial_point = [0.0, 0.5, 0.25, 0.75]

# Fit specified model to historical data
f = fit!(
    gas, y;
    initial_params = initial_params,
    opt_method = IPNewton(gas, [initial_point]; ub=ub, lb=lb)
)

# Print estimation statistics
fit_stats(f)
