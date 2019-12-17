using ScoreDrivenModels, Statistics, DelimitedFiles

const SDM = ScoreDrivenModels

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

y = readdlm("./test/data/BG96.csv")[:]

# Evaluate some approximate initial params
initial_params = [mean(y) var(y)]

# Define upper bounds and lower bounds for parameters
ub = [1.0; 1.0; 1.0; 1.0]
lb = [-1.0; 0.0; 0.0; 0.0]

# Define a GAS model where only \sigma^2 varys over time
gas = GAS(1, 1, Normal, 1.0, time_varying_params = [2])

# Estimate the model
res = estimate!(gas, y; initial_params = initial_params, opt_method = IPNewton(gas, 50; ub = ub, lb = lb))