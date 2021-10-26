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

GPU = false

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
Iv = Integral((vx,vy,vz) in DomainSets.ProductDomain(ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf)))

# Equations
E = [Dx(Φ(t,x,y,z)), Dy(Φ(t,x,y,z)), Dz(Φ(t,x,y,z))]
Divx_v = vx * Dx(f(t,x,y,z,vx,vy,vz)) + vy * Dy(f(t,x,y,z,vx,vy,vz)) + vz * Dz(f(t,x,y,z,vx,vy,vz))
F = e/m_e .* E
DfDv = [Dvx(f(t,x,y,z,vx,vy,vz)), Dvy(f(t,x,y,z,vx,vy,vz)), Dvz(f(t,x,y,z,vx,vy,vz))]
Divv_F = dot(F, DfDv)
∇²Φ = Dx(DxΦ(t,x,y,z)) + Dy(DyΦ(t,x,y,z)) + Dz(DzΦ(t,x,y,z))

eqs = [Dt(f(t,x,y,z,vx,vy,vz)) ~ - Divx_v - Divv_F
       ∇²Φ ~ e/ε_0 * Dt(f(t,x,y,z,vx,vy,vz))]

der_ = [Dx(Φ(t,x,y,z)) ~ DxΦ(t,x,y,z),
        Dy(Φ(t,x,y,z)) ~ DyΦ(t,x,y,z),
        Dz(Φ(t,x,y,z)) ~ DzΦ(t,x,y,z)]

# Boundaries and initial conditions
bcs_ = [f(0,x,y,z,vx,vy,vz) ~ set_initial_geometry(x,y,z) * (π*v_th^2)^(-3/2) * exp(-((vx + vy + vz)/3)^2/(v_th^2)), # Maxwellian for now averaging 3 components of
       Φ(0,x,y,z) ~ set_initial_geometry(x,y,z) * e*n_0/ε_0 * Dt(f(0,x,y,z,vx,vy,vz))]

bcs__ = [bcs_;der_]

# Neural Network
CUDA.allowscalar(false)
chain = [FastChain(FastDense(7, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1));
         [FastChain(FastDense(4, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)) for _ in 1:4]]
initθ = GPU ? map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain)) : map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) 

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
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=200)
phi = discretization.phi

ts, xs, ys, zs, vxs, vys, vzs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

###### Possible solution
u_predict = collect(Array(phi[1]([t,x,y,z,vx,vy,vz], minimizers_[1]))[1] for t in ts, x in xs, y in ys, z in zs, vx in vxs, vy in vys, vz in vzs)

u_predict_x

u_predict_x = u_predict[:,:,:,:,1,1,1]
u_predict_v = u_predict[:,1,1,1,:,:,:]

anim = @animate for t ∈ eachindex(ts)
    p1 = scatter(u_predict_x[t,:,1,1], u_predict_x[t,1,:,1], u_predict_x[t,1,1,:])
    p2 = scatter(u_predict_v[t,:,1,1], u_predict_v[t,1,:,1], u_predict_v[t,1,1,:])
    plot(p1,p2)
end
gif(anim,"result.gif", fps=10)

res.u # are the weights of the neural Network

@show phi[1]

phi[1](0.2, minimizers_[1])
phi[2](0.3, minimizers_[2])


u_predict_test = [reshape([first(phi[2]([t,x,y,z],minimizers_[2])) for x in xs  for y in ys for z in zs], (length(xs),length(ys), length(zs)))  for t in ts]
anim = @animate for i=1:length(ts)
    p1 = plot(xs, ys, u_real[i], st=:surface, title = "real");
    p2 = plot(xs, ys, u_predict[i], st=:surface,title = "predict, t = $(ts[i])");
    plot(p1,p2)
  end

# discretization.phi gives the internal representation of u(x, y). We can use it to visualize a solution via