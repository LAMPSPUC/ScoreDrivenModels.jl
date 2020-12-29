module ScoreDrivenModels

import Base: length, deepcopy, show

using Distributions
using Optim
using SpecialFunctions
using LinearAlgebra
using Printf

const SCALINGS  = [0.0, 1/2, 1.0]
const BIG_NUM = 1e8
const SMALL_NUM = 1e-8

abstract type ScoreDrivenModel end

end
