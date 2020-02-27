"""
    Location Scale Student's t

* Parametrization
parametrized in \\mu, \\sigma^2 and \\nu

* Score

* Fisher Information

* `time_varying_params` map.

* Default link
"""
LocationScaleTDist

function score!(score_til::Matrix{T}, y::T, ::Type{LocationScaleTDist}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = ((param[t, 3] + 1) * (y - param[t, 1])) / ((y - param[t, 1])^2 + param[t, 2] * param[t, 3])
    score_til[t, 2] = -(param[t, 3] * (param[t, 2] - (y - param[t, 1])^2)) / (2 * param[t, 2] * (param[t, 3] * param[t, 2] + 
                        (y - param[t, 1])^2))
    score_til[t, 3] = ((((y - param[t, 1])^2)*(param[t, 3] + 1)/(param[t, 3] * (y - param[t, 1])^2 + param[t, 2] * param[t, 3]^2)) -
                      1/param[t, 3] - 
                      log(((y - param[t, 1])^2)/(param[t, 2] * param[t, 3]) + 1) - 
                      digamma(param[t, 3] / 2) + digamma((param[t, 3] + 1) / 2)) / 2
    return
end

function fisher_information!(aux::AuxiliaryLinAlg{T}, ::Type{LocationScaleTDist}, param::Matrix{T}, t::Int) where T
    aux.fisher[1, 1] = (param[t, 3] + 1.0)/(param[t, 2] * (param[t, 3] + 3.0))
    aux.fisher[1, 2] = zero(T)
    aux.fisher[2, 1] = zero(T)
    aux.fisher[1, 3] = zero(T)
    aux.fisher[3, 1] = zero(T)
    aux.fisher[2, 2] = param[t, 3] / ((2 * param[t, 2]^2) * (param[t, 3] + 3.0))
    aux.fisher[2, 3] = -1.0 / (param[t, 2] * (param[t, 3] + 3.0)*(param[t, 3] + 1.0))
    aux.fisher[3, 2] = -1.0 / (param[t, 2] * (param[t, 3] + 3.0)*(param[t, 3] + 1.0))
    aux.fisher[3, 3] = 0.5 * (0.5 * trigamma(0.5 * param[t, 3]) - 0.5 * trigamma( 0.5 * (param[t, 3] + 1.0) ) -
                       (param[t, 3] + 5.0) / (param[t, 3] * (param[t, 3] + 3.0) * (param[t, 3] + 1.0)))
    return
end

function log_likelihood(::Type{LocationScaleTDist}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = zero(T)
    for t in 1:n
        loglik -= 0.5 * log(param[t, 2]*param[t, 3]) + logbeta(0.5, param[t, 3]/2) + ((param[t, 3] + 1)/2) *
                  log(1 + (1/(param[t, 3]*param[t, 2])) * (y[t] - param[t, 1])^2)
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{LocationScaleTDist}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(IdentityLink, param[t, 1])
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
    param_tilde[t, 3] = link(LogLink, param[t, 3], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{LocationScaleTDist}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(IdentityLink, param_tilde[t, 1])
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
    param[t, 3] = unlink(LogLink, param_tilde[t, 3], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryLinAlg{T}, ::Type{LocationScaleTDist}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(IdentityLink, param[t, 1])
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    aux.jac[3] = jacobian_link(LogLink, param[t, 3], zero(T))
    return
end

# utils 
function update_dist(::Type{LocationScaleTDist}, param::Matrix{T}, t::Int) where T
    tdist = TDist(param[t, 3])
    return LocationScale(param[t, 1], sqrt(param[t, 2]), tdist)
end 

function num_params(::Type{LocationScaleTDist})
    return 3
end