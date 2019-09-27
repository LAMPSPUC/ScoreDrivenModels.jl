"""
Proof somewhere 
parametrized in \\mu and \\sigma^2
"""
function score(y::T, ::Type{Normal}, param::Vector{T}) where T
    return [
        (y - param[1])/param[2];
        -(0.5/param[2]) * (1 - ((y - param[1])^2)/param[2])
    ]
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
function log_likelihood(::Type{Normal}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = zero(T)
    for i in 1:n
        loglik -=  0.5*(log(2*pi*param[i][2]) + (1/param[i][2])*(y[i] - param[i][1])^2)
    end
    return -loglik
end

# Links
function param_to_param_tilde(::Type{Normal}, param::Vector{T}) where T 
    return [
        param_to_param_tilde(IdentityLink, param[1]);
        param_to_param_tilde(ExponentialLink, param[2])
    ]
end
function param_tilde_to_param(::Type{Normal}, param_tilde::Vector{T}) where T 
    return [
        param_tilde_to_param(IdentityLink, param_tilde[1]);
        param_tilde_to_param(ExponentialLink, param_tilde[2])
    ]
end
function jacobian_param_tilde(::Type{Normal}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_param_tilde(IdentityLink, param_tilde[1]);
        jacobian_param_tilde(ExponentialLink, param_tilde[2])
    ])
end

# utils 
function update_dist(D::Type{Normal}, param::Vector{T}) where T
    # normal here is parametrized as sigma^2
    return Normal(param[1], sqrt(param[2]))
end 

function num_params(::Type{<:Normal})
    return 2
end