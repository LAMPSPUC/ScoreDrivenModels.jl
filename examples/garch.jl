using ScoreDrivenModels, Statistics, DelimitedFiles

const SDM = ScoreDrivenModels

# The GARCH(1, 1) is equivaent to a GAS(1, 1) Normal model with inverse scaling. The models are only equivalent under 
# no reparametrizations, here this means that the you should fit the model overwriting the default link!, unlink! and
# jacobian_link! methods for the Normal distribution to use only the IdentityLink.

# Overwrite link interface to work with Identity links.
function SDM.link!(param_tilde::Matrix{T}, ::Type{Normal}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(IdentityLink, param[t, 1])
    param_tilde[t, 2] = link(IdentityLink, param[t, 2])
    return
end
function SDM.unlink!(param::Matrix{T}, ::Type{Normal}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(IdentityLink, param_tilde[t, 1])
    param[t, 2] = unlink(IdentityLink, param_tilde[t, 2])
    return
end
function SDM.jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{Normal}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(IdentityLink, param[t, 1])
    aux.jac[2] = jacobian_link(IdentityLink, param[t, 2])
    return
end

# Data from [Bollerslev and Ghysels (JBES 1996)](https://doi.org/10.2307/1392425). Took from ARCHModels.jl
y = readdlm("./test/data/BG96.csv")[:]

# Evaluate some approximate initial params
initial_params = [mean(y) var(y)]

# There are 2 possibilities to estimate the model.
# The first one is to understand that in the GARCH estimation process usually one 
# defines that the parameters ω, α_0, α_1 and β_1 are constrained. The problem is 
# that the equivalent GAS model relies that 
# ω_1 = ω
# ω_2 = α_0
# A_1 = α_1
# B_1 - A_1 = β_1
# Here there is no easy way of adding a constraint B_1 - A_1 >= 0. An alternative is to define 
# bounds assuring the upper bound of A_1 is smaller than the lower bound for B_1.

# Define upper bounds and lower bounds for parameters ω_1, ω_2, A_1, B_1
ub = [1.0; 1.0; 0.5; 1.0]
lb = [-1.0; 0.0; 0.0; 0.5]

# Define a GAS model where only \sigma^2 is varying over time
gas = GAS(1, 1, Normal, 1.0, time_varying_params = [2])

# Give an initial_point in the interior of the bounds.
initial_point = [0.0; 0.5; 0.25; 0.75]

# Estimate the model
f = fit(gas, y; initial_params = initial_params, 
                  opt_method = IPNewton(gas, [initial_point]; ub = ub, lb = lb))

# Another way to estimate the model is to rely on luck and give many random initial_points
# for a non constrained optimization 
f = fit(gas, y; initial_params = initial_params, 
                  opt_method = NelderMead(gas, 100))