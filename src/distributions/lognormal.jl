"""
Proof somewhere 
parametrized in \\mu and \\sigma^2
"""
function score(y::T, ::Type{LogNormal}, param::Vector{T}) where T
    return [
        (log(y) - param[1])/param[2] ;
        -(0.5/param[2]) * (1 - ((log(y) - param[1])^2)/param[2])
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
    loglik = -0.5*log(2*pi)*n
    for i in 1:n
        loglik -= log(y[i]*sqrt(param[i][2])) + 0.5*(log(y[i]) - param[i][1])^2/param[i][2]
    end
    return -loglik
end

# Links
function unlink(::Type{LogNormal}, param_tilde::Vector{T}) where T 
    return [
        unlink(IdentityLink, param_tilde[1]);
        unlink(LogLink, param_tilde[2], zero(T))
    ]
end
function jacobian_link(::Type{LogNormal}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_link(IdentityLink, param_tilde[1]);
        jacobian_link(LogLink, param_tilde[2], zero(T))
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