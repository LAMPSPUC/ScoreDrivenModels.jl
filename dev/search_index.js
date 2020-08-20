var documenterSearchIndex = {"docs":
[{"location":"examples/#Examples-1","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#GARCH(1,-1)-1","page":"Examples","title":"GARCH(1, 1)","text":"","category":"section"},{"location":"examples/#","page":"Examples","title":"Examples","text":"GARCH (acronym for generalized autoregressive conditional heteroskedasticity) is a well-known time series model used to describe the conditional variance of the errors in time-varying fashion. In this example, we show how the GARCH(1, 1) is simply a particular case of GAS; more specifically, it is a Normal GAS(1, 1) model with inverse scaling.","category":"page"},{"location":"examples/#","page":"Examples","title":"Examples","text":"Note that this equivalence is true only under no reparametrization – it is necessary to modify the default link!, unlink! and jacobian_link! methods to use IdentityLink for the Normal distribution.","category":"page"},{"location":"examples/#","page":"Examples","title":"Examples","text":"The code for this example can be accessed here.","category":"page"},{"location":"examples/#Water-inflow-1","page":"Examples","title":"Water inflow","text":"","category":"section"},{"location":"examples/#","page":"Examples","title":"Examples","text":"Let's model some monthly water inflow data from the Northeast of Brazil using a Lognormal GAS model. Since inflow is a highly seasonal phenomenon, we will utilize lags 1 and 12. The former aims to characterize the short-term evolution of the series, while the latter characterizes the seasonality. The full code is in the examples folder.","category":"page"},{"location":"examples/#","page":"Examples","title":"Examples","text":"# Convert data to vector\ny = Vector{Float64}(vec(inflow'))\n\n# Specify model: here we use lag 1 for trend characterization and \n# lag 12 for seasonality characterization\ngas = Model([1, 12], [1, 12], LogNormal, 0.0)\n\n# Estimate the model via MLE\nfit!(gas, y)\n\n# Obtain in-sample estimates for the inflow\ny_gas = fitted_mean(gas, y, dynamic_initial_params(y, gas))\n\n# Compare observations and in-sample estimates\nplot(y)\nplot!(y_gas)","category":"page"},{"location":"examples/#","page":"Examples","title":"Examples","text":"The result can be seen in the following plot.","category":"page"},{"location":"examples/#","page":"Examples","title":"Examples","text":"(Image: Historical inflow data vs. in-sample estimates)","category":"page"},{"location":"manual/#Manual-1","page":"Manual","title":"Manual","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"For a concise overview of the package, please read the package paper.  It provides a high-level presentation of the theory behind score-driven models and showcases the features of the package as well as examples.","category":"page"},{"location":"manual/#Model-Specification-1","page":"Manual","title":"Model Specification","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"Model","category":"page"},{"location":"manual/#ScoreDrivenModels.Model","page":"Manual","title":"ScoreDrivenModels.Model","text":"Model\n\nThe constructor of a score-driven model. The model receives the lag structure, the  distribution and the scaling. You can define the lag structure in two different  ways, either by passing integers p and q to add all lags from 1 to p and 1 to q or by  passing vectors of integers ps and qs containing the desired lags. Once you build  the model all of the unknown parameters that must be estimated are represented as NaN.\n\n# Passing p and q\njulia> Model(2, 2, LogNormal, 0.5)\nModel{LogNormal,Float64}([NaN, NaN], Dict(2=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), Dict(2=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), 0.5)\n\n# Passing ps and qs\njulia> Model([1, 12], [1, 12], Gamma, 0.0)\nModel{Gamma,Float64}([NaN, NaN], Dict(12=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), Dict(12=>[NaN 0.0; 0.0 NaN],1=>[NaN 0.0; 0.0 NaN]), 0.0)\n\nIf you don't want all the parameters to be considered time-varying you can express it  through the keyword argument time_varying_params, there you should pass a vector containing a number that represents which parameter should be time-varying. As an example in the Normal distribution time_varying_params = [1] indicates that only mu is time-varying. You can find the table with the dictionary (number => parameter) in the section ScoreDrivenModels distributions.\n\njulia> Model([1, 12], [1, 12], Normal, 1.0; time_varying_params = [1])\nModel{Normal,Float64}([NaN, NaN], Dict(12=>[NaN 0.0; 0.0 0.0],1=>[NaN 0.0; 0.0 0.0]), Dict(12=>[NaN 0.0; 0.0 0.0],1=>[NaN 0.0; 0.0 0.0]), 1.0)\n\n\n\n\n\n","category":"type"},{"location":"manual/#Optimization-Algorithms-1","page":"Manual","title":"Optimization Algorithms","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.jl allows users to use different optimization methods, in particular it has a common interface to easily incorporate algorithms available on Optim.jl","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"All optimization methods can receive the following keyword arguments","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"f_tol - Relative tolerance in changes of the objective value. Default is 1e-6.\ng_tol - Absolute tolerance in the gradient, in infinity norm. Default is 1e-6.\niterations - Maximum number of iterations. Default is 10^5.\nLB - Lower bound of the initial points. Default is 0.0.\nUB - Upper bound of the initial points. Default is 0.6.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.IPNewton allows users to perform box-constrained optimization.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.NelderMead\nScoreDrivenModels.LBFGS\nScoreDrivenModels.IPNewton","category":"page"},{"location":"manual/#ScoreDrivenModels.NelderMead","page":"Manual","title":"ScoreDrivenModels.NelderMead","text":"NelderMead(model::Model, args...; kwargs...)\n\nIf an Int is provided the method will sample that many random initial_points and use them as initial  points for Optim NelderMead method. If a Vector{Vector{T}} is provided it will use them as  initial points for Optim NelderMead method.\n\n\n\n\n\n","category":"type"},{"location":"manual/#ScoreDrivenModels.LBFGS","page":"Manual","title":"ScoreDrivenModels.LBFGS","text":"LBFGS(model::Model, args...; kwargs...)\n\nIf an Int is provided the method will sample that many random initial_points and use them as  initial points for Optim LBFGS method. If a Vector{Vector{T}} is provided it will use them as  initial points for Optim LBFGS method.\n\n\n\n\n\n","category":"type"},{"location":"manual/#ScoreDrivenModels.IPNewton","page":"Manual","title":"ScoreDrivenModels.IPNewton","text":"IPNewton(model::Model, args...; kwargs...)\n\nIf an Int is provided the method will sample that many random initial_points and use them as  initial points for Optim IPNewton method. If a Vector{Vector{T}} is provided it will use them as  initial points for Optim IPNewton method.\n\nThis method provides an alternative to create box constraints. constraints can be passed as a Vector the default constraints are ub = Inf * ones(T, dim_unknowns(model)) and lb = -Inf * ones(T, dim_unknowns(model))\n\n\n\n\n\n","category":"type"},{"location":"manual/#Recursion-1","page":"Manual","title":"Recursion","text":"","category":"section"},{"location":"manual/#Links-1","page":"Manual","title":"Links","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"Links are reparametrizations utilized to ensure certain parameter is within its original domain.  For instance, for a particular distribution, one might want to ensure that the time varying  parameter is positive: f in mathbbR^+. The way to do this is to model tildef = lnf.  More generally, one can establish that tildef = h(f). We refer to this procedure as  linking. When a parameter is linked, the GAS recursion happens in the domain of tildef  and then one can recover the original parameter by f = left(hright)^-1(tilde f).  We refer to this procedure as unlinking. The new recursion becomes:","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"beginequation*leftbeginarrayccl\n    f_t = h^-1(tilde f_t) \n    tilde f_t+1 = omega + sum_i=1^p A_itilde s_t-i+1 + sum_j=1^q B_jtilde f_t-j+1\n    endarray\n    right\nendequation*","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"Notice that a different parametrization alters the dynamics of the model. For example,  the GAS(1,1) model with Normal distribution and scaling d = 1 is equivalent to the well-known  GARCH(1, 1) model. Conversely, if a different parametrization is utilized, the model will  no longer be equivalent.","category":"page"},{"location":"manual/#Types-of-links-1","page":"Manual","title":"Types of links","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"The abstract type Link subsumes any type of link that can be expressed.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.IdentityLink\nScoreDrivenModels.LogLink\nScoreDrivenModels.LogitLink","category":"page"},{"location":"manual/#ScoreDrivenModels.IdentityLink","page":"Manual","title":"ScoreDrivenModels.IdentityLink","text":"IdentityLink <: Link\n\nDefine the map tildef = f where f in mathbbR and tildef in mathbbR\n\n\n\n\n\n","category":"type"},{"location":"manual/#ScoreDrivenModels.LogLink","page":"Manual","title":"ScoreDrivenModels.LogLink","text":"LogLink <: Link\n\nDefine the map tildef = ln(f - a) where f in a infty) a in mathbbR and tildef in mathbbR\n\n\n\n\n\n","category":"type"},{"location":"manual/#ScoreDrivenModels.LogitLink","page":"Manual","title":"ScoreDrivenModels.LogitLink","text":"LogitLink <: Link\n\nDefine the map tildef = ln(fracf - ab - f) where f in a b a b in mathbbR and tildef in mathbbR\n\n\n\n\n\n","category":"type"},{"location":"manual/#Link-functions-1","page":"Manual","title":"Link functions","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.link\nScoreDrivenModels.unlink\nScoreDrivenModels.jacobian_link","category":"page"},{"location":"manual/#ScoreDrivenModels.link","page":"Manual","title":"ScoreDrivenModels.link","text":"link(args...)\n\nThe link function is a map that brings a parameter f in a subspace mathcalF subset mathbbR to mathbbR.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.unlink","page":"Manual","title":"ScoreDrivenModels.unlink","text":"unlink(args...)\n\nThe unlink function is the inverse map of link. It brings tildef in mathbbR to the subspace mathcalF subset mathbbR.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.jacobian_link","page":"Manual","title":"ScoreDrivenModels.jacobian_link","text":"jacobian_link(args...)\n\nEvaluates the derivative of the link with respect to the parameter f.\n\n\n\n\n\n","category":"function"},{"location":"manual/#Forecasting-1","page":"Manual","title":"Forecasting","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.jl allows users to generate point forecasts, confidence intervals  forecasts or ensembles of scenarios. Point forecasts are obtained using the function forecast  and ensembles of scenarios are obtained using the function simulate.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"forecast_quantiles\nsimulate","category":"page"},{"location":"manual/#ScoreDrivenModels.simulate","page":"Manual","title":"ScoreDrivenModels.simulate","text":"simulate(series::Vector{T}, gas::Model{D, T}, H::Int, S::Int, kwargs...) where {D, T}\n\nGenerate scenarios for the future of a time series by updating the GAS recursion H times and taking a sample of the distribution until it generates S scenarios.\n\nBy default this method uses the stationary_initial_params method to perform the  score driven recursion. If you estimated the model with a different set of initial_params use them here to maintain the coherence of your estimation.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels-distributions-1","page":"Manual","title":"ScoreDrivenModels distributions","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"The following section presents how every distribution is parametrized, its score, Fisher information and the time_varying_params map. Every distribution is originally imported to ScoreDrivenModels.jl from Distributions.jl. Some distributions may have different parametrizations from Distributions.jl, this is handled internally.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"Distribution Identity scaling Inverse and inverse square root scalings\nBeta ✓ ✓\nBetaLocationScale ✓ x\nExponential ✓ ✓\nGamma ✓ ✓\nLogitNormal ✓ ✓\nLogNormal ✓ ✓\nNegativeBinomial ✓ x\nNormal ✓ ✓\nPoisson ✓ ✓\nTDist ✓ ✓\nTDistLocationScale ✓ ✓\nWeibull ✓ x","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.Beta\nScoreDrivenModels.BetaLocationScale\nScoreDrivenModels.Exponential\nScoreDrivenModels.Gamma\nScoreDrivenModels.LogitNormal\nScoreDrivenModels.LogNormal\nScoreDrivenModels.NegativeBinomial\nScoreDrivenModels.Normal\nScoreDrivenModels.Poisson\nScoreDrivenModels.TDist\nScoreDrivenModels.TDistLocationScale\nScoreDrivenModels.Weibull","category":"page"},{"location":"manual/#Distributions.Beta","page":"Manual","title":"Distributions.Beta","text":"Beta\n\nParametrization\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#ScoreDrivenModels.BetaLocationScale","page":"Manual","title":"ScoreDrivenModels.BetaLocationScale","text":"BetaLocationScale\n\nParametrization \\mu, \\sigma, \\alpha, \\beta\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.Exponential","page":"Manual","title":"Distributions.Exponential","text":"Exponential\n\nParametrization\n\nparametrized in \\lambda\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.Gamma","page":"Manual","title":"Distributions.Gamma","text":"Gamma\n\nParametrization\n\nparametrized in \\alpha and \\theta\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.LogitNormal","page":"Manual","title":"Distributions.LogitNormal","text":"LogitNormal\n\nParametrization\n\nparametrized in \\mu and \\sigma^2\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.LogNormal","page":"Manual","title":"Distributions.LogNormal","text":"LogNormal\n\nParametrization\n\nparametrized in \\mu and \\sigma^2\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.NegativeBinomial","page":"Manual","title":"Distributions.NegativeBinomial","text":"NegativeBinomial\n\nParametrization\n\nparametrized in r, p\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.Normal","page":"Manual","title":"Distributions.Normal","text":"Normal\n\nParametrization\n\nparametrized in \\mu and \\sigma^2\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.Poisson","page":"Manual","title":"Distributions.Poisson","text":"Poisson\n\nParametrization\n\nparametrized in \\lambda\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.TDist","page":"Manual","title":"Distributions.TDist","text":"Student's t\n\nParametrization\n\nparametrized in \\nu\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#ScoreDrivenModels.TDistLocationScale","page":"Manual","title":"ScoreDrivenModels.TDistLocationScale","text":"Student's t location scale\n\nParametrization\n\nparametrized in \\mu, \\sigma^2 and \\nu\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Distributions.Weibull","page":"Manual","title":"Distributions.Weibull","text":"LogNormal\n\nParametrization\n\nparametrized in \\alpha and \\theta\n\nScore\nFisher Information\ntime_varying_params map.\nDefault link\n\n\n\n\n\n","category":"type"},{"location":"manual/#Implementing-a-new-distribution-1","page":"Manual","title":"Implementing a new distribution","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"If you want to add a new distribution please feel free to make a pull request.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"Each distribution must have the following methods:","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.score!\nScoreDrivenModels.fisher_information!\nScoreDrivenModels.log_likelihood\nlink interface\nScoreDrivenModels.link!\nScoreDrivenModels.unlink!\nScoreDrivenModels.jacobian_link!\nScoreDrivenModels.update_dist\nScoreDrivenModels.params_sdm\nScoreDrivenModels.num_params","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"The details of the new distribution must be documented following the example in Normal and added to the ScoreDrivenModels distributions section. The new implemented distribution must also be added to the constant DISTS and exported in the distributions/common_interface.jl file.","category":"page"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.score!\nScoreDrivenModels.fisher_information!\nScoreDrivenModels.log_likelihood\nScoreDrivenModels.link!\nScoreDrivenModels.unlink!\nScoreDrivenModels.jacobian_link!\nScoreDrivenModels.update_dist\nScoreDrivenModels.params_sdm\nScoreDrivenModels.num_params","category":"page"},{"location":"manual/#ScoreDrivenModels.score!","page":"Manual","title":"ScoreDrivenModels.score!","text":"score!(score_til::Matrix{T}, y::T, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T\n\nFill score_til with the score of distribution D with parameters param[:, t] considering the observation y.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.fisher_information!","page":"Manual","title":"ScoreDrivenModels.fisher_information!","text":"fisher_information!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T\n\nFill aux with the fisher information of distribution D with parameters param[:, t].\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.log_likelihood","page":"Manual","title":"ScoreDrivenModels.log_likelihood","text":"log_likelihood(D::Type{<:Distribution}, y::Vector{T}, param::Matrix{T}, n::Int) where T\n\nEvaluate the log-likelihood of the distribution D considering the time-varying parameters param and the observations y.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.link!","page":"Manual","title":"ScoreDrivenModels.link!","text":"link!(param_tilde::Matrix{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T\n\nFill param_tilde after the unlinking procedure of param.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.unlink!","page":"Manual","title":"ScoreDrivenModels.unlink!","text":"unlink!(param::Matrix{T}, D::Type{<:Distribution}, param_tilde::Matrix{T}, t::Int) where T\n\nFill param after the unlinking procedure of param_tilde.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.jacobian_link!","page":"Manual","title":"ScoreDrivenModels.jacobian_link!","text":"jacobian_link!(aux::AuxiliaryLinAlg{T}, D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T\n\nWrite the jacobian of the link map in aux.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.update_dist","page":"Manual","title":"ScoreDrivenModels.update_dist","text":"update_dist(D::Type{<:Distribution}, param::Matrix{T}, t::Int) where T\n\nCreate a new distribution from Distributions.jl based on the parametrization used in ScoreDrivenModels.jl.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.params_sdm","page":"Manual","title":"ScoreDrivenModels.params_sdm","text":"params_sdm(d::Distribution)\n\nRecover the parametrization used in ScoreDrivenModels.jl based on a Distribution from Distributions.jl.\n\n\n\n\n\n","category":"function"},{"location":"manual/#ScoreDrivenModels.num_params","page":"Manual","title":"ScoreDrivenModels.num_params","text":"num_params(D::Type{<:Distribution})\n\nNumber of parameters of a given distribution.\n\n\n\n\n\n","category":"function"},{"location":"manual/#Reference-1","page":"Manual","title":"Reference","text":"","category":"section"},{"location":"manual/#","page":"Manual","title":"Manual","text":"ScoreDrivenModels.Unknowns","category":"page"},{"location":"manual/#ScoreDrivenModels.Unknowns","page":"Manual","title":"ScoreDrivenModels.Unknowns","text":"Unknowns\n\nStructure that stores the positions of the parameters to be estimated in the Model.\n\n\n\n\n\n","category":"type"},{"location":"#ScoreDrivenModels.jl-Documentation-1","page":"Home","title":"ScoreDrivenModels.jl Documentation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"ScoreDrivenModels.jl is a Julia package for modeling, forecasting, and simulating time  series with score-driven models, also known as generalized autoregressive score models (GAS).  Implementations are based on the paper  Generalized Autoregressive Models with Applications  by D. Creal, S. J. Koopman, and A. Lucas.","category":"page"},{"location":"#Installation-1","page":"Home","title":"Installation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This package is registered so you can simply add it using Julia's Pkg manager:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"pkg> add ScoreDrivenModels","category":"page"},{"location":"#Citing-the-package-1","page":"Home","title":"Citing the package","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"If you use ScoreDrivenModels.jl in your work, we kindly ask you to cite the package paper:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"@misc{bodin2020scoredrivenmodelsjl,\ntitle={ScoreDrivenModels.jl: a Julia Package for Generalized Autoregressive Score Models},\nauthor={Guilherme Bodin and Raphael Saavedra and Cristiano Fernandes and Alexandre Street},\nyear={2020},\neprint={2008.05506}\n}","category":"page"},{"location":"#Contributing-1","page":"Home","title":"Contributing","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Contributions to this package are more than welcome, if you find a bug or have any suggestions  for the documentation please post it on the  github issue tracker.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"When contributing please note that the package follows the  JuMP style guide.","category":"page"}]
}
