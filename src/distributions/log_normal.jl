"""
Proof somewhere 
parametrized in \\mu and \\sigma^2
"""
function score(y::T, ::Type{LogNormal}, param::Vector{T}) where T
    return [
        (log(y) - param[1])/param[2] ;
        7 #TODO
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Type{LogNormal}, param::Vector{T}) where T
    Diagonal([1/(param[2]); 1/(2*(param[2]^2))])
end

"""
Proof somewhere
"""
function log_likelihood(::Type{LogNormal}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in 1:n
        loglik += 1#TODO
    end
    return -loglik
end

# Links
function param_to_param_tilde(::Type{LogNormal}, param::Vector{T}) where T 
    return [
        param_to_param_tilde(IdentityLink, param[1]);
        param_to_param_tilde(ExponentialLink, param[2])
    ]
end
function param_tilde_to_param(::Type{LogNormal}, param_tilde::Vector{T}) where T 
    return [
        param_tilde_to_param(IdentityLink, param_tilde[1]);
        param_tilde_to_param(ExponentialLink, param_tilde[2])
    ]
end
function jacobian_param_tilde(::Type{LogNormal}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_param_tilde(IdentityLink, param_tilde[1]);
        jacobian_param_tilde(ExponentialLink, param_tilde[2])
    ])
end

# utils 
function update_dist(::Type{LogNormal}, param::Vector{T}) where T
    # lognormal here is parametrized as sigma^2
    return LogNormal(param[1], sqrt(param[2]))
end 

function num_params(::Type{LogNormal})
    return 2
end