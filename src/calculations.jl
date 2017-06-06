# likelihood

logl(w, Ldx) = sum(log(Ldx*w))
dlogl(w, Ldx) = sum(Ldx./(Ldx*w), 1) |> vec


# entropy

repeatweights(w, zmult::Int) = repmat(w/zmult, zmult)
repeatweights(w, L::Matrix)  = repeatweights(w, Int(size(L,1) / length(w)))

# Monte Carlo approximation to the z-entropy
function hz(w, L, wz=repeatweights(w, L))
  rhoz = L * w           # \Int L(z|x) * pi(x) dx
  l = 0.
  for (r,w) in zip(rhoz, wz)
    r == 0 && continue
    l -= log(r)*w
  end
  l
end

# derivative of the MC approximation hz
function dhz(w, L, wz=repeatweights(w, L))
  nz, nx  = size(L)
  zmult   = Int(nz/nx)

  rhoz    = L * w
  factors = wz ./ rhoz

  d = zeros(w)
  @Threads.threads for k = 1:nx
    @simd for i = 1:nz
      @inbounds d[k] -= L[i,k] * factors[i]
    end
    @simd for m = 0:zmult-1
      @inbounds d[k] -= log(rhoz[k+nx*m]) / zmult
    end
  end
  d
end
