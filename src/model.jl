type LikelihoodModel{X, Y}
  xs :: Vector{X}
  ys :: Vector{Y}
  zs :: Vector{Y}
  datas :: Vector{Y}
  Ldx :: Matrix
  Lzx :: Matrix
end 


# constructors

LikelihoodModel(xs, ys, zs, datas, measerr) = 
  LikelihoodModel(xs, ys, zs, datas, likelihoodmat(datas, ys, measerr), likelihoodmat(zs, ys, measerr))

function LikelihoodModel(; xs=[], phi=Void, ys=[], datas=[], measerr=Void, prior=Void, ndata=0, zs=[], zmult=0, sfact=0, smoothkernel=Void)

  if ys == [] && phi != Void
    ys = phi.(xs)
  end

  if datas == [] && ndata > 0
    datas = phi.(rand(prior, ndata)) + rand(measerr, ndata)
  end

  if zs == [] && zmult > 0
    zs = repmat(ys, zmult) + rand(measerr, length(ys)*zmult)
  end

  if sfact > 0
    datas = smoothdata(smoothkernel, sfact, datas)
    measerr = convolute(measerr, smoothkernel)
  end

  m = LikelihoodModel(xs, ys, zs, datas, measerr)
end

# likelihood computation

likelihoodmat(xs, ys, d::Distribution) = map(a->pdf(d, a), (x-y for x in xs, y in ys))

# model wrappers

npmle(m)     = mple(m, 0)
refprior(m)  = mple(m, 1)

logl(m::LikelihoodModel, w) = logl(w, m.Ldx)
dlogl(m::LikelihoodModel, w) = dlogl(w, m.Ldx) 

hz(m::LikelihoodModel, w)  =  hz(w, m.Lzx)
dhz(m::LikelihoodModel, w) = dhz(w, m.Lzx)

function mple_obj(m::LikelihoodModel, reg)
  if reg == 0
    w -> logl(m, w)
  elseif reg == 1
    w -> hz(m, w)
  else
    w -> reg*hz(m,w) + (1-reg) * logl(m,w)
  end
end

function dmple_obj(m::LikelihoodModel, reg)
  if reg == 0
    w -> dlogl(m, w)
  elseif reg == 1
    w -> dhz(m,w)
  else
    w -> reg*dhz(m, w) + (1-reg) * dlogl(m, w)
  end
end
