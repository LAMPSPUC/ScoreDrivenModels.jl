"""
Proof somewhere 
parametrized in \\mu and \\sigma^2
"""
function score!(score_til::Matrix{T}, y::T, ::Type{Normal}, param::Matrix{T}, t::Int) where T
    score_til[t, 1] = (y - param[t, 1])/param[t, 2]
    score_til[t, 2] = -(0.5/param[t, 2]) * (1 - ((y - param[t, 1])^2)/param[t, 2])
    return
end

"""
Proof somewhere
"""
function fisher_information(::Type{Normal}, param::Vector{T}) where T
    return Diagonal([1/(param[2]); 1/(2*(param[2]^2))])
end

"""
Proof:
p = 1/sqrt(2πσ²) exp(-0.5(y-μ)²/σ²)

ln(p) = -0.5ln(2πσ²)-0.5(y-μ)²/σ²
"""
function log_likelihood(::Type{Normal}, y::Vector{T}, param::Matrix{T}, n::Int) where T
    loglik = -0.5 * n * log(2 * pi)
    for t in 1:n
        loglik -= 0.5 * (log(param[t, 2]) + (1 / param[t, 2]) * (y[t] - param[t, 1]) ^ 2)
    end
    return -loglik
end

# Links
function link!(param_tilde::Matrix{T}, ::Type{Normal}, param::Matrix{T}, t::Int) where T 
    param_tilde[t, 1] = link(IdentityLink, param[t, 1])
    param_tilde[t, 2] = link(LogLink, param[t, 2], zero(T))
    return
end
function unlink!(param::Matrix{T}, ::Type{Normal}, param_tilde::Matrix{T}, t::Int) where T 
    param[t, 1] = unlink(IdentityLink, param_tilde[t, 1])
    param[t, 2] = unlink(LogLink, param_tilde[t, 2], zero(T))
    return
end
function jacobian_link!(aux::AuxiliaryStruct{T}, ::Type{Normal}, param::Matrix{T}, t::Int) where T 
    aux.jac[1] = jacobian_link(IdentityLink, param[t, 1])
    aux.jac[2] = jacobian_link(LogLink, param[t, 2], zero(T))
    return
end

# utils 
function update_dist(::Type{Normal}, param::Matrix{T}, t::Int) where T
    # normal here is parametrized as sigma^2
    return Normal(param[t, 1], sqrt(param[t, 2]))
end 

function num_params(::Type{Normal})
    return 2
end