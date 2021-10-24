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

# Plot
ts, xs, ys, zs, vxs, vys, vzs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers_ = [res.minimizer[s] for s in sep]

anim_Φ = @animate for t ∈ ts
    @info "Animating frame t..."
    u_predict_Φ = reshape([phi[2]([t,x,y,z], minimizers_[2])[1] for x in xs for y in ys for z in zs], length(xs), length(ys), length(zs))
    #p1 = plot(xs, ys, zs, u_predict_E, label="", title="E")
    p1 = plot(xs,ys,zs, line_z = u_predict_Φ)
    plot(p1)
end
gif(anim_Φ,"Phi.gif", fps=10)

anim_f_x = @animate for t ∈ ts
    @info "Animating frame t..."
    u_predict_f = reshape([phi[1]([t,x,y,z,vx,vy,vz], minimizers_[1])[1] for x in xs for y in ys for z in zs for vx in vxs for vy in vys for vz in vzs], length(xs), length(ys), length(zs),length(vxs), length(vys), length(vzs))
    p2 = plot(xs, ys, zs, line_z=u_predict_f, label="", title="f_x")
    plot(p2)
end
gif(anim_f_x,"f_x.gif", fps=10)


u_predict = collect(Array(phi[1]([t,x,y,z,vx,vy,vz], minimizers_[1]))[1] for x in xs, y in ys, z in zs)



anim_f_v = @animate for t ∈ ts
    @info "Animating frame t..."
    u_predict_f = reshape([phi[1]([t,x,y,z,vx,vy,vz], minimizers_[1])[1] for x in xs for y in ys for z in zs for vx in vxs for vy in vys for vz in vzs], length(xs), length(ys), length(zs),length(vxs), length(vys), length(vzs))
    p2 = plot(vxs, vys, vzs, line_z=u_predict_f, label="", title="f_v")
    plot(p2)
end
gif(anim_f_v,"f_v.gif", fps=10)


function plot_3D(phi, res, initθ,  model_name="")
    ts, xs, ys, zs, vxs, vys, vzs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
    acum =  [0;accumulate(+, length.(initθ))]
    sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
    minimizers_ = [res.minimizer[s] for s in sep]

    
end

function plot_E_1D(phi, minimizers_, model_name="")
    anim = @animate for t ∈ ts
        @info "Animating frame t..."
        u_predict_E = reshape([phi[2]([t,x], minimizers_[2])[1] for x in xs], length(xs))
        p1 = plot(xs, u_predict_E, label="", title="E")
        plot(p1)
    end
    gif(anim,model_name*"E.gif", fps=10)
end

function plot_f_1D(phi, minimizers_, model_name="")
    anim = @animate for t ∈ ts
        @info "Animating frame t..."
        u_predict_f = reshape([phi[1]([t,x,v], minimizers_[1])[1] for x in xs for v in vs], length(xs), length(vs))
        p1 = plot(xs, vs, u_predict_f, st=:surface, label="", title="f")
        plot(p1)
    end
    gif(anim,model_name*"f.gif", fps=10)
end