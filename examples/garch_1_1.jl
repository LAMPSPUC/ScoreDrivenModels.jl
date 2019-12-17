using ScoreDrivenModels

const SDM = ScoreDrivenModels

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

y = BG96
using Statistics
ini = [mean(y) var(y)]
ub = [1.0; 1.0; 1.0; 1.0]
lb = [-1.0; 0.0; 0.0; 0.0]
gas = GAS(1, 1, Normal, 1.0, time_varying_params = [2])
res = estimate!(gas, BG96; initial_params = ini, opt_method = IPNewton(gas, 50; ub = ub, lb = lb))
res.coefs
res.llk
loglikelihood(f)
c = coef(f)

using Test
@test gas.ω[1] - -0.00616637237701241 ≈ 0 atol = 1e-4 rtol = 1e-4
@test gas.ω[2] - 0.010760592759725487 ≈ 0 atol = 1e-4 rtol = 1e-4
@test gas.A[1][2, 2] - 0.15341133818189595 ≈ 0 atol = 1e-4 rtol = 1e-4
@test gas.B[1][2, 2] - (0.15341133818189595 + 0.8058745318161223) ≈ 0 atol = 1e-4 rtol = 1e-4