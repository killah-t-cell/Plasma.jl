
cb = function (p,l)
       println("Current loss is: $l")
       return false
   end

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
Iv = Integral(v in DomainSets.ClosedInterval(-100, 100)) 

# Geometry
function ic_geometry(x)
	if (x < 0.3 && x > 0.1) 1 else 0. end
end
@register ic_geometry(x)

# Equations
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
   
# Solve
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=1000)