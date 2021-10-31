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
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=2)
phi = discretization.phi

ts, xs, ys, zs, vxs, vys, vzs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

###### Possible solution
u_predict = [collect(Array(phi[1]([t,x,y,z,vx,vy,vz], minimizers_[1]))[1] for x in xs, y in ys, z in zs, vx in vxs, vy in vys, vz in vzs) for t in ts]

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
    p1 = plot(xs, ys, u_predict_test[i], st=:surface, title = "real");
    plot(p1)
  end

# discretization.phi gives the internal representation of u(x, y). We can use it to visualize a solution via
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x, y, z
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C=1
eq  = Dtt(u(t,x,y,z)) ~ C^2*(Dxx(u(t,x,y,z))+Dyy(u(t,x,y,z))+Dzz(u(t,x,y,z)))

# Initial and boundary conditions
bcs = [u(t,0, y, z) ~ 0.,# for all t > 0
       u(t,1, y, z) ~ 0.,# for all t > 0
       u(t,x, 0, z) ~ 0.,# for all t > 0
       u(t,x, 1, z) ~ 0.,# for all t > 0
       u(t,x, y, 0) ~ 0.,# for all t > 0
       u(t,x, y, 1) ~ 0.,# for all t > 0
       u(0,x, y, z) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(0,x, y, z)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0),
           y ∈ Interval(0.0,1.0),
           z ∈ Interval(0.0,1.0)]
# Discretization
dx = 0.1

# Neural network
chain = FastChain(FastDense(4,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
discretization = PhysicsInformedNN(chain, GridTraining(dx); init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[t,x,y,z],[u(t,x,y,z)])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# optimizer
opt = BFGS()
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=500)
phi = discretization.phi
sol.u([0,0,0,0])
sol.w([0,1,2,3])
sol.u(domains)

### More elegant way to retrieve PINN
ts, xs, ys, zs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
u_predict = [collect(phi([t,x,y,z], res.minimizer)[1] for x in xs, y in xs, z in zs) for t in ts]


# I can plot it like so
anim = @animate for t ∈ eachindex(ts)
    scatter(u_predict[t])
end
gif(anim, "wave3d.gif", fps=10)

# I can plot y against x, y
anim = @animate for t ∈ eachindex(xs)
    plot(xs,ys,u_predict[t][1,:,1])
end
gif(anim, "wave3d.gif", fps=10) 

using Makie

positions = Node(u_predict[1])
scena = volume(xs, ys, zs, positions, colormap = :plasma, colorrange = (minimum(vol), maximum(vol)),figure = (; resolution = (800,800)),  
                axis = (; type=Axis3, perspectiveness = 0.5,  azimuth = 7.19, elevation = 0.57,  
                aspect = (1,1,1)))

fps = 60
record(scena, "output.mp4", eachindex(ts)) do t
    positions[] = u_predict[t]
    sleep(1/fps)
end



## CORRECT WAY TO DO 7D PLOTTING!!!!

using GLMakie
GLMakie.activate!()

ts = 1:10
data = [rand(100,100,100,100,100,100) for t in ts]

i = Observable(1)

f = Figure()

data_1 = @lift(data[$i][:, :, :, 1, 1, 1])
data_2 = @lift(data[$i][1, 1, 1, :, :, :])

GLMakie.volume(f[1, 1], data_1, axis = (;type = Axis3))
GLMakie.volume(f[1, 2], data_2, axis = (;type = Axis3))

ls = labelslider!(f, "t", ts)
f[2, 1:2] = ls.layout
connect!(i, ls.slider.value)

f

# real data

using GLMakie
GLMakie.activate!()

ts = 1:10
i = Observable(1)

u_predict

f = Figure()

data_1 = @lift(u_predict[$i][:, :, :, 1, 1, 1])
data_2 = @lift(u_predict[$i][1, 1, 1, :, :, :])

GLMakie.volume(f[1, 1], data_1, axis = (;type = Axis3))
GLMakie.volume(f[1, 2], data_2, axis = (;type = Axis3))

ls = labelslider!(f, "t", ts)
f[2, 1:2] = ls.layout
connect!(i, ls.slider.value)

f
