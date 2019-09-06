abstract type Link end

struct PoissonLink <: Link end

h(::PoissonLink, λ::Real) = log(λ)
invh(::PoissonLink, λ::Real) = exp(λ)

