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
    dMu = param[1]
    dPhi2 = param[2]
    dNu = param[3]

    dNu_s = digamma((dNu + 1.0)/2.0)*0.5 - digamma(dNu/2.0)*0.5 - 1.0/(2.0*dNu) - 0.5*log(1.0+ (y-dMu)^2.0/(dNu*dPhi2) )+
    0.5*(dNu+1.0)*(y-dMu)^2.0/(dPhi2*(dNu)^2.0 )/(1.0 +  (y-dMu)^2.0/(dNu*dPhi2) );

    dPhi2_s = -1.0/(2.0*dPhi2) + (dNu+1.0)*(y-dMu)^2.0/(2.0*(dPhi2)^2.0*dNu*(1.0 +  (y-dMu)^2.0/(dNu*dPhi2)));

    dMu_s = (dNu+1.0)*(y-dMu)/(dNu*dPhi2 + (y-dMu)^2.0);
    return [
        dMu_s;
        dPhi2_s;
        dNu_s
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Type{LocationScale{Float64,TDist{Float64}}}, param::Vector{T}) where T
    dPhi2 = param[2]
    dNu = param[3]

    uu = (dNu+1.0)/(dPhi2*(dNu+3.0));
    dd = dNu/(2.0*(dPhi2^2.0)*(dNu+3.0));
    tt = 0.5*( 0.5* trigamma(0.5*dNu) - 0.5* trigamma( 0.5*(dNu+1.0) ) - (dNu+5.0)/(dNu*(dNu+3.0)*(dNu+1.0)));
    td = -1.0/(dPhi2*(dNu+3.0)*(dNu+1.0));

    mIM=zeros(3,3);

    mIM[1,1]=uu;
    mIM[2,2]=dd;
    mIM[3,3]=tt;
    mIM[3,2]=td;
    mIM[2,3]=td;

    return mIM
end

"""
Proof somewhere
"""
function log_likelihood(::Type{LocationScale{Float64,TDist{Float64}}}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in 1:n
        loglik +=   log(gamma((param[i][3]+1)/2)/(gamma(param[i][3]/2)*sqrt(pi*param[i][3]))) + 
                    0.5*log(1/param[i][2])+
                    ((param[i][3]+1)/2)*log(1+(1/param[i][3])*((y[i]-param[i][1])^2)/param[i][2])
    end
    return -loglik
end


# utils 
function update_dist(::Type{LocationScale{Float64,TDist{Float64}}}, param::Vector{T}) where T
    # Generalized Stundet T here is parametrized as sigma^2
    return LocationScale{Float64,TDist{Float64}}(param[1], sqrt(param[2]), TDist(param[3]))
end 

function num_params(::Type{LocationScale{Float64,TDist{Float64}}})
    return 3
end

##############################################
# Still TODO
##############################################

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