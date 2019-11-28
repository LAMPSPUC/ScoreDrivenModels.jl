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
    loglik = -0.5*n*log(2*pi)
    for i in 1:n
        loglik -= 0.5*(log(param[i][2]) + (1/param[i][2])*(y[i] - param[i][1])^2)
    end
    return -loglik
end

# Links
function link(::Type{Normal}, param::Vector{T}) where T 
    return [
        link(IdentityLink, param[1]);
        link(LogLink, param[2], zero(T))
    ]
end
function unlink(::Type{Normal}, param_tilde::Vector{T}) where T 
    return [
        unlink(IdentityLink, param_tilde[1]);
        unlink(LogLink, param_tilde[2], zero(T))
    ]
end
function jacobian_link(::Type{Normal}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_link(IdentityLink, param_tilde[1]);
        jacobian_link(LogLink, param_tilde[2], zero(T))
    ])
end

# utils 
function update_dist(::Type{Normal}, param::Vector{T}) where T
    # normal here is parametrized as sigma^2
    return Normal(param[1], sqrt(param[2]))
end 

function num_params(::Type{Normal})
    return 2
end