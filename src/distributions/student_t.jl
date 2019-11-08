##############################################
# Generalized Student's t
# Distribution.jl: LocationScale(μ,σ_2,TDist(ν))
# parametrized in μ, σ^2 and ν
# wikipedia: https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution
##############################################
const GeneralTDist = LocationScale{Float64,TDist{Float64}}

"""
Proof somewhere 
parametrized in μ, σ_2 and ν
"""
function score(y::T, ::Type{GeneralTDist}, param::Vector{T}) where T
    μ = param[1]
    σ_2 = param[2]
    ν = param[3]

    ν_s = digamma((ν + 1.0)/2.0)*0.5 - digamma(ν/2.0)*0.5 - 1.0/(2.0*ν) - 0.5*log(1.0+ (y-μ)^2.0/(ν*σ_2) )+
    0.5*(ν+1.0)*(y-μ)^2.0/(σ_2*(ν)^2.0 )/(1.0 +  (y-μ)^2.0/(ν*σ_2) );

    σ_2_s = -1.0/(2.0*σ_2) + (ν+1.0)*(y-μ)^2.0/(2.0*(σ_2)^2.0*ν*(1.0 +  (y-μ)^2.0/(ν*σ_2)));

    μ_s = (ν+1.0)*(y-μ)/(ν*σ_2 + (y-μ)^2.0);
    return [
        μ_s;
        σ_2_s;
        ν_s
    ]
end

"""
Proof somewhere
"""
function fisher_information(::Type{GeneralTDist}, param::Vector{T}) where T
    σ_2 = param[2]
    ν = param[3]

    uu = (ν+1.0)/(σ_2*(ν+3.0));
    dd = ν/(2.0*(σ_2^2.0)*(ν+3.0));
    tt = 0.5*( 0.5* trigamma(0.5*ν) - 0.5* trigamma( 0.5*(ν+1.0) ) - (ν+5.0)/(ν*(ν+3.0)*(ν+1.0)));
    td = -1.0/(σ_2*(ν+3.0)*(ν+1.0));

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
function log_likelihood(::Type{GeneralTDist}, y::Vector{T}, param::Vector{Vector{T}}, n::Int) where T
    loglik = 0.0
    for i in 1:n
        loglik +=   lgamma((param[i][3]+1)/2)-lgamma(param[i][3]/2)-log(sqrt(pi*param[i][3])) + 
                    0.5*log(1/param[i][2])+
                    ((param[i][3]+1)/2)*log(1+(1/param[i][3])*((y[i]-param[i][1])^2)/param[i][2])
    end
    return -loglik
end


# utils 
function update_dist(::Type{GeneralTDist}, param::Vector{T}) where T
    # Generalized Student's t is parametrized here as σ^2
    return LocationScale(param[1], sqrt(param[2]), TDist(param[3]))
end 

function νm_params(::Type{GeneralTDist})
    return 3
end

# Links
function param_to_param_tilde(::Type{GeneralTDist}, param::Vector{T}) where T 
    return [
        param_to_param_tilde(IdentityLink, param[1]);
        param_to_param_tilde(ExponentialLink, param[2], zero(T));
        param_to_param_tilde(ExponentialLink, param[3], one(T))
    ]
end
function param_tilde_to_param(::Type{GeneralTDist}, param_tilde::Vector{T}) where T 
    return [
        param_tilde_to_param(IdentityLink, param_tilde[1]);
        param_tilde_to_param(ExponentialLink, param_tilde[2], zero(T));
        param_tilde_to_param(ExponentialLink, param_tilde[3], one(T))
    ]
end
function jacobian_param_tilde(::Type{GeneralTDist}, param_tilde::Vector{T}) where T 
    return Diagonal([
        jacobian_param_tilde(IdentityLink, param_tilde[1]);
        jacobian_param_tilde(ExponentialLink, param_tilde[2], zero(T));
        jacobian_param_tilde(ExponentialLink, param_tilde[3], one(T))
    ])
end