using ScoreDrivenModels, Statistics, DelimitedFiles

# Overwrite link interface to use IdentityLink for Normal distribution.
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

y = vec(readdlm("./test/data/BG96.csv"));
initial_params = [mean(y) var(y)];
ub = [1.0; 1.0; 0.5; 1.0];
lb = [-1.0; 0.0; 0.0; 0.5];
gas = Model(1, 1, Normal, 1.0, time_varying_params = [2]);
initial_point = [0.0; 0.5; 0.25; 0.75];
f = fit!(gas, y; initial_params = initial_params,
                opt_method = IPNewton(gas, [initial_point]; ub = ub, lb = lb));
estimation_stats = fit_stats(f)