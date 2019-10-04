##############################################
# Generalized Student T distribution
# Distribution.jl: LocationScale(Mu,Phi2,TDist(Nu))
# parametrized in \\mu, \\sigma^2 and \\nu
# wikipedia: https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution
##############################################
using SpecialFunctions

"""
Proof somewhere 
parametrized in \\mu, \\sigma^2 and \\nu
"""

function score(y::T, ::Type{LocationScale{Float64,TDist{Float64}}}, param::Vector{T}) where T
    dNu = param[1]
    dPhi2 = param[2]
    dMu = param[2]

    dNu_s = digamma((dNu + 1.0)/2.0)*0.5 - digamma(dNu/2.0)*0.5 - 1.0/(2.0*dNu) - 0.5*log(1.0+ (y-dMu)^2.0/(dNu*dPhi2) )+
    0.5*(dNu+1.0)*(y-dMu)^2.0/(dPhi2*(dNu)^2.0 )/(1.0 +  (y-dMu)^2.0/(dNu*dPhi2) );

    dPhi2_s = -1.0/(2.0*dPhi2) + (dNu+1.0)*(y-dMu)^2.0/(2.0*(dPhi2)^2.0*dNu*(1.0 +  (y-dMu)^2.0/(dNu*dPhi2)));

    dMu_s = (dNu+1.0)*(y-dMu)/(dNu*dPhi2 + (y-dMu)^2.0);
    return [
        dNu_s;
        dPhi2_s;
        dMu_s
    ]
end
##############################################
# Still TODO
##############################################

"""
Proof somewhere
"""
function fisher_information(::Type{LocationScale{Float64,TDist{Float64}}}, param::Vector{T}) where T
    return Diagonal([1/(param[2]); 1/(2*(param[2]^2))])
end

"""
Proof:
p = 1/sqrt(2πσ²) exp(-0.5(y-μ)²/σ²)

ln(p) = -0.5ln(2πσ²)-0.5(y-μ)²/σ²
"""
function log_likelihood(::Type{LocationScale{Float64,TDist{Float64}}}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = -0.5*n*log(2*pi)
    for i in 1:n
        loglik -= 0.5*(log(param[i][2]) + (1/param[i][2])*(y[i] - param[i][1])^2)
    end
    return -loglik
end

# Links
function param_to_param_tilde(::Type{LocationScale{Float64,TDist{Float64}}}, param::Vector{T}) where T 
    return [
        param_to_param_tilde(IdentityLink, param[1]);
        param_to_param_tilde(ExponentialLink, param[2])
    ]
end
function param_tilde_to_param(::Type{LocationScale{Float64,TDist{Float64}}}, param_tilde::Vector{T}) where T 
    return [
        param_tilde_to_param(IdentityLink, param_tilde[1]);
        param_tilde_to_param(ExponentialLink, param_tilde[2])
    ]
end
function jacobian_param_tilde(::Type{LocationScale{Float64,TDist{Float64}}}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_param_tilde(IdentityLink, param_tilde[1]);
        jacobian_param_tilde(ExponentialLink, param_tilde[2])
    ])
end

# utils 
function update_dist(::Type{LocationScale{Float64,TDist{Float64}}}, param::Vector{T}) where T
    # Generalized Stundet T here is parametrized as sigma^2
    return LocationScale{Float64,TDist{Float64}}(param[1], sqrt(param[2]))
end 

function num_params(::Type{LocationScale{Float64,TDist{Float64}}})
    return 3
end