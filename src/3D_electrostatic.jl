# abstracted to 3 dimensions
using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
using DomainSets
import ModelingToolkit: Interval, infimum, supremum
using Plots
using LinearAlgebra
using CUDA

@parameters t x y z vx vy vz
@variables f(..) Φ(..)
@variables DxΦ(..),DyΦ(..),DzΦ(..)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2
Dvx = Differential(vx)
Dvy = Differential(vy)
Dvz = Differential(vz)
Dt = Differential(t)

# Geometry
function set_initial_geometry(x, y, z)
	if (x > 0.2 && x < 0.3) && (y > 0.2 && y < 0.3) && (z > 0.2 && z < 0.3) 1 else 0. end
end
@register set_initial_geometry(x,y,z)

# Constants
μ_0 = 1.25663706212e-6 # N A⁻²
ε_0 = 8.8541878128e-12 # F ms⁻¹
e   = 1.602176634e-19 # Coulombs
m_e = 9.10938188e-31 # Kg
n_0 = 1
Kb = 1.3806503e-23
T = 30000
v_th = sqrt(2 * Kb * T/m_e)

# Space
domains = [t ∈ Interval(0.0, 1.0),
           x ∈ Interval(0.0, 1.0),
           y ∈ Interval(0.0, 1.0), 
           z ∈ Interval(0.0, 1.0), 
           vx ∈ Interval(0.0, 1.0),
           vy ∈ Interval(0.0, 1.0),
           vz ∈ Interval(0.0, 1.0)]

# Integrals
Iv = Integral((vx,vy,vz) in DomainSets.ProductDomain(ClosedInterval(0 ,1), ClosedInterval(0 ,1), ClosedInterval(0 ,1)))
Ix = Integral((x,y,z) in DomainSets.ProductDomain(ClosedInterval(0 ,1), ClosedInterval(0 ,1), ClosedInterval(0 ,1)))

# Equations
E = [Dx(Φ(t,x,y,z)), Dy(Φ(t,x,y,z)), Dz(Φ(t,x,y,z))]
Divx_v = Dx(vx * f(t,x,y,z,vx,vy,vz)) + Dy(vy * f(t,x,y,z,vx,vy,vz)) + Dz(vz * f(t,x,y,z,vx,vy,vz))
Divv_F = Dx(e/m_e * E[1] * f(t,x,y,z,vx,vy,vz)) + Dy(e/m_e * E[2] * f(t,x,y,z,vx,vy,vz)) + Dz(e/m_e * E[3] * f(t,x,y,z,vx,vy,vz))
∇²Φ = Dx(DxΦ(t,x,y,z)) + Dy(DyΦ(t,x,y,z)) + Dz(DzΦ(t,x,y,z))

eqs = [Dt(f(t,x,y,z,vx,vy,vz)) ~ - Divx_v - Divv_F
       ∇²Φ ~ e/ε_0 * Iv(f(t,x,y,z,vx,vy,vz))]

der_ = [Dx(Φ(t,x,y,z)) ~ DxΦ(t,x,y,z),
        Dy(Φ(t,x,y,z)) ~ DyΦ(t,x,y,z),
        Dz(Φ(t,x,y,z)) ~ DzΦ(t,x,y,z)]

# Boundaries and initial conditions
bcs_ = [f(0,x,y,z,vx,vy,vz) ~ set_initial_geometry(x,y,z) * (π*v_th^2)^(-3/2) * exp(-((vx + vy + vz)/3)^2/(v_th^2)), # Maxwellian for now averaging 3 components of velocity
       Φ(0,x,y,z) ~ set_initial_geometry(x,y,z) * e*n_0/ε_0 * Iv(f(0,x,y,z,vx,vy,vz))] # we may need to change this to the analytical Green's solution

bcs__ = [bcs_;der_]

# Neural Network
chain = [FastChain(FastDense(7, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1));
         [FastChain(FastDense(4, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)) for _ in 1:4]]
initθ = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) # initθ = map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain))

discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params=initθ)
vars = [f(t,x,y,z,vx,vy,vz), Φ(t,x,y,z), DxΦ(t,x,y,z), DyΦ(t,x,y,z), DzΦ(t,x,y,z)]
@named pde_system = PDESystem(eqs, bcs__, domains, [t,x,y,z,vx,vy,vz], vars)
prob = SciMLBase.symbolic_discretize(pde_system, discretization)
prob = SciMLBase.discretize(pde_system, discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# Solve
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=1000)
phi = discretization.phi

