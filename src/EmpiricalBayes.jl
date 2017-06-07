module EmpiricalBayes

using Distributions
using NLopt

include("model.jl")
include("optim.jl")
include("calculations.jl")


export LikelihoodModel
export npmle, mple, refprior
export OptConfig


end
