using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
using DomainSets
import ModelingToolkit: Interval, infimum, supremum
using Plots
using CUDA

@parameters t x v
@variables f(..) E(..) 
Dx = Differential(x)
Dt = Differential(t)
Dv = Differential(v)

# Constants
μ_0 = 1.25663706212e-6 # N A⁻²
ε_0 = 8.8541878128e-12 # F ms⁻¹
e   = 1.602176634e-19 # Coulombs
m_e = 9.10938188e-31 # Kg
n_0 = 1
v_th = sqrt(2)

# Integrals
Iv = Integral(v in DomainSets.ClosedInterval(-1, 1)) 

eqs = [Dt(f(t,x,v)) ~ - v * Dx(f(t,x,v)) - e/m_e * E(t,x) * Dv(f(t,x,v))
       Dx(E(t,x)) ~ e*n_0/ε_0 * (Iv(f(t,x,v)) - 1)]

bcs = [f(0,x,v) ~ 1/(v_th * sqrt(2π)) * exp(-v^2/(2*v_th^2)),
       E(0,x) ~ e*n_0/ε_0 * (Iv(f(0,x,v)) - 1)]

domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0), 
           v ∈ Interval(0.0, 1.0)]

# Neural Network
chain = [FastChain(FastDense(3, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)),
         FastChain(FastDense(2, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1))]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params= initθ)
@named pde_system = PDESystem(eqs, bcs, domains, [t,x,v], [f(t,x,v), E(t,x)])
prob = SciMLBase.symbolic_discretize(pde_system, discretization)
prob = SciMLBase.discretize(pde_system, discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# Solve
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=5)
phi = discretization.phi

# Plot
ts, xs, vs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

function plot_f(res)
    anim = @animate for t ∈ ts
        @info "Animating frame t..."
        u_predict_f = reshape([phi[1]([t,x,v], minimizers_[1])[1] for x in xs for v in vs], length(xs), length(vs))
        p1 = plot(xs, vs, u_predict_f, st=:surface, label="", title="f")
        plot(p1)
    end
    gif(anim,"f.gif", fps=10)
end

function plot_E(res)
    anim = @animate for t ∈ ts
        @info "Animating frame t..."
        u_predict_E = reshape([phi[2]([t,x], minimizers_[2])[1] for x in xs], length(xs))
        p1 = plot(xs, u_predict_E, label="", title="E")
        plot(p1)
    end
    gif(anim,"E.gif", fps=10)
end

plot_E(res)
plot_f(res)