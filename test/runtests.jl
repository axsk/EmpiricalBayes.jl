using EmpiricalBayes
using Distributions

m=LikelihoodModel(xs=collect(linspace(0,1,100)), phi=(x->x), measerr=Normal(0,.1), ndata=5, prior=Uniform(0,1), zmult = 10)

npmle(m)
refprior(m)
@time mple(m, 0.5)
@time mple(m, 0.5, config=OptConfig(METHOD=:auglag))
@time EmpiricalBayes.mpb(m, .5)
#@time EmpiricalBayes.jump(m, 0.5)

#em(m, 1000)
# plots
