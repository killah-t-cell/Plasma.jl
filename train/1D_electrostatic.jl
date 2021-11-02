using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
using DomainSets
import ModelingToolkit: Interval, infimum, supremum
using Plots
using CUDA

GPU = false

@parameters t x v
@variables f(..) Ivf(..) E(..)
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

# Space
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0), 
           v ∈ Interval(0.0, 1.0)]

# Integrals
Iv = Integral(v in DomainSets.ClosedInterval(-Inf, Inf)) 

# Equations
eqs = [Dt(f(t,x,v)) ~ - v * Dx(f(t,x,v)) - e/m_e * E(t,x) * Dv(f(t,x,v))
       Dx(E(t,x)) ~ e*n_0/ε_0 * (Ivf(t,x,v) - 1)]

# Boundaries and initial conditions
function set_initial_geometry(v)
	if (v > 0.2 && v < 0.3) 1 else 0. end
end
@register set_initial_geometry(v)

bcs_ = [f(0,x,v) ~ set_initial_geometry(v) * 1/(v_th * sqrt(2π)) * exp(-v^2/(2*v_th^2)),
       E(0,x) ~ set_initial_geometry(v) * e*n_0/ε_0 * (Ivf(t,x,v) - 1) * x,
       E(t,0) ~ 0]

ints_ = [Iv(f(t,x,v)) ~ Ivf(t,x,v)]

bcs = [bcs_;ints_]

# Neural Network
CUDA.allowscalar(false)
chain = [FastChain(FastDense(3, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)),
         FastChain(FastDense(3, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)),
         FastChain(FastDense(2, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1))]

initθ = GPU ? map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain)) : map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) 


discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params= initθ)
@named pde_system = PDESystem(eqs, bcs, domains, [t,x,v], [f(t,x,v), Ivf(t,x,v), E(t,x)])
prob = SciMLBase.symbolic_discretize(pde_system, discretization)
prob = SciMLBase.discretize(pde_system, discretization)

pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents
bcs_inner_loss_functions = inner_loss_functions[1:3]
aprox_integral_loss_functions = inner_loss_functions[4:end]

cb = function (p,l)
    println("loss: ", l )
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    println("int_losses: ", map(l_ -> l_(p), aprox_integral_loss_functions))
    return false
end

# Solve
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=200)
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, ADAM(0.01), cb = cb, maxiters=10000)
prob = remake(prob, u0=res.minimizer)
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=200)
phi = discretization.phi

