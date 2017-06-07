using NLopt
using Parameters

@with_kw type OptConfig
    LOWERXB = 1e-12
    XTOLREL = 1e-5
    XTOLABS = 0
    FTOLREL = 1e-5
    CTOLABS = 1e-5
    DEBUG   = false
    MAXEVAL = 0
    METHOD = :ineq
    OPTIMIZER = :LD_MMA
end

uniformweights(m::LikelihoodModel) = uniformweights(length(m.xs))
uniformweights(n) = fill(1/n, n)

function mple(m, reg, w0=uniformweights(m); config=OptConfig())
    @unpack_OptConfig config

    # generate closures
    f  = mple_obj(m,reg)
    df = dmple_obj(m,reg)
    
    function objective(x,g)
        # note the sign due to minimizing solver
        if length(g) == n
            g[:] = -df(x)
        end
        fx = f(x)
        DEBUG && @printf("f: f(x)=%f sum(x)=%f \n", fx, sum(x))
        -f(x)
    end

    local opt

    n = length(w0)
    opt = Opt(OPTIMIZER, n)

    min_objective!(opt, objective)
    lower_bounds!(opt, LOWERXB)
    upper_bounds!(opt, 1)

    xtol_rel!(opt, XTOLREL)
    xtol_abs!(opt, XTOLABS)
    ftol_rel!(opt, FTOLREL)

    maxeval!(opt, MAXEVAL)

    if METHOD == :ineq
        myineq(x, g)  = ((length(g) == n) && (g[:] = -1); 
                         sum(x) - 1. - CTOLABS/2)
        myineq2(x, g) = ((length(g) == n) && (g[:] = -1); 
                         1. - sum(x) - CTOLABS/2)

        inequality_constraint!(opt, myineq, CTOLABS/2)
        inequality_constraint!(opt, myineq2, CTOLABS/2)

    elseif METHOD == :auglag

        function eqconst(x, grad)
            fill!(grad, 1)
            sum(x) - 1
        end

        # store inner optimizer and overwrite with outer
        inneropt = opt
        opt  = Opt(:LD_AUGLAG, n)

        min_objective!(opt, objective)
        lower_bounds!(opt, LOWERXB)
        upper_bounds!(opt, 1)

        equality_constraint!(opt, eqconst, CTOLABS)


        xtol_rel!(opt, XTOLREL)
        xtol_abs!(opt, XTOLABS)
        ftol_rel!(opt, FTOLREL)

        local_optimizer!(opt, inneropt)
    end

    minf, minx, ret = optimize(opt, w0)

    if minx == w0
        warn("Optimizer returned initial proposal")
    end

    minx
end

using JuMP
function jump(m,reg)
    n = length(m.xs)
    jm=Model()
    @variable(jm, 0 <= x[1:n] <= 1)

    f  = mple_obj(m,reg)
    df = dmple_obj(m,reg)

    JuMP.register(jm, :obj, n, (x...)->f(x), (g,x...)->(g[:] = df(x)))

    @constraint(jm, sum(x) == 1)
    @eval @NLobjective($jm, Max, $(Expr(:call, :obj, [Expr(:ref,:x, i) for i=1:n]...)))

    solve(jm)
end

using MathProgBase

type StochasticProblem <: MathProgBase.AbstractNLPEvaluator
    n
    f
    df
end

SP  = StochasticProblem
MPB = MathProgBase

# initialization
MPB.initialize(d::SP, rf) = ()
MPB.features_available(d::SP) = [:Grad, :Jac]

# f
MPB.eval_f(d::SP, x) = d.f(x)
MPB.eval_grad_f(d::SP, g, x) = g[:]=d.df(x)

# constraints
MPB.eval_g(d::SP, g, x) = g[1]=sum(x)
MPB.jac_structure(d::SP) = ones(Int,d.n), collect(1:d.n)
MPB.eval_jac_g(d::SP, J, x) = J[:]=ones(Int, d.n)


MPB.isobjlinear(d::SP) = false
MPB.isobjquadratic(d::SP) = false
MPB.isconstrlinear(d::SP) = true

using Ipopt

function mpb(m, reg)
    nlm = MPB.NonlinearModel(IpoptSolver())
    n = length(m.xs)
    sp = SP(n, mple_obj(m,reg), dmple_obj(m,reg))
    MPB.loadproblem!(nlm, n, 1, fill(0, n), fill(1, n), [1], [1], :Max, sp)
    MPB.setwarmstart!(nlm, uniformweights(n))
    MPB.optimize!(nlm)
    @show MathProgBase.status(nlm)
end
