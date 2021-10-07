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
using BSON: @save
using BSON: @load

function set_initial_geometry(dim)

end

function maxwellian_3D(vx,vy,vz,T,m, v_drift)
    Kb = 1.3806503e-23
    v_th = sqrt(2*Kb*T/m)
    v = sqrt(vx^2 + vy^2 + vz^2)
    v_drift = sqrt(v_drift[1]^2 + v_drift[2]^2 + v_drift[3]^2)
    return (π*v_th^2)^(-3/2) * exp(-(v - v_drift)^2/v_th^2)
end

function maxwellian_1D(v,T,v_th, v_drift)
    Kb = 1.3806503e-23
    v_th = sqrt(2*Kb*T/m)
    return (π*v_th^2)^(-3/2) * exp(-(v - v_drift)^2/v_th^2)
end

function solve_collisionless_plasma(geometry::Function, lb, ub, T, ic_fe, ic_fi=ic_fe, v_drift=zeros(3), time_lb=lb, time_ub=ub, GPU=true; args...)
    if lb > ub
        error("lower bound must be larger than upper bound")
    end

    if ic_fe(args...) < 0 || ic_fi(args...) < 0
        error("distribution function must be greater than 0")
    end

    @parameters t x y z vx vy vz
    @variables fe(..) fi(..) ϕ(..) Ax(..) Ay(..) Az(..)
    @variables Dxϕ(..) Dyϕ(..) Dzϕ(..) Dtϕ(..) DxAx(..) DyAx(..) DzAx(..) DtAx(..) DxAy(..) DyAy(..) DzAy(..) DtAy(..) DxAz(..) DyAz(..) DzAz(..) DtAz(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    Dvx = Differential(vx)
    Dvy = Differential(vy)
    Dvz = Differential(vz)
    Dt = Differential(t)

    # Registrations
    @register geometry(x,y,z)
    @register ic_fe(vx,vy,vz,T,m, v_drift)
    @register ic_fi(vx,vy,vz,T,m, v_drift)

    # Constants
    μ_0 = 1.25663706212e-6 # N A⁻²
    ε_0 = 8.8541878128e-12 # F ms⁻¹
    q_e   = 1.602176634e-19 # Coulombs
    q_i   = 1.602176634e-19 # Coulombs
    m_e = 9.10938188e-31 # Kg
    m_i = 33435837724e-27 # Deuterium mass

    # Space
    domains = [t ∈ Interval(time_lb, time_ub),
            x ∈ Interval(lb, ub),
            y ∈ Interval(lb, ub), 
            z ∈ Interval(lb, ub), 
            vx ∈ Interval(lb, ub),
            vy ∈ Interval(lb, ub),
            vz ∈ Interval(lb, ub)]

    # Integrals
    Iv = Integral((vx,vy,vz) in DomainSets.ProductDomain(ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf)))

    # Equations
    curl(vec) = [Dy(vec[3]) - Dz(vec[2]), Dz(vec[1]) - Dx(vec[3]), Dx(vec[2]) - Dy(vec[1])]
    A = [Ax(t,x,y,z), Ay(t,x,y,z), Az(t,x,y,z)]
    E = [-Dx(ϕ(t,x,y,z))-Dt(Ax(t,x,y,z)),-Dy(ϕ(t,x,y,z))-Dt(Ay(t,x,y,z)), -Dz(ϕ(t,x,y,z))-Dt(Ay(t,x,y,z))]
    B = curl(A)

    v = [vx, vy, vz]
    Divx_v_e = vx * Dx(fe(t,x,y,z,vx,vy,vz)) + vy * Dy(fe(t,x,y,z,vx,vy,vz)) + vz * Dz(fe(t,x,y,z,vx,vy,vz))
    Divx_v_i = vx * Dx(fi(t,x,y,z,vx,vy,vz)) + vy * Dy(fi(t,x,y,z,vx,vy,vz)) + vz * Dz(fi(t,x,y,z,vx,vy,vz))
    F_e = q_e/m_e * (E + cross(v, B))
    F_i = q_i/m_i * (E + cross(v, B))
    DfDv_e = [Dvx(fe(t,x,y,z,vx,vy,vz)), Dvy(fe(t,x,y,z,vx,vy,vz)), Dvz(fe(t,x,y,z,vx,vy,vz))]
    DfDv_i = [Dvx(fi(t,x,y,z,vx,vy,vz)), Dvy(fi(t,x,y,z,vx,vy,vz)), Dvz(fi(t,x,y,z,vx,vy,vz))]
    Divv_F_e = dot(F_e, DfDv_e)
    Divv_F_i = dot(F_i, DfDv_i)

    ρ = q_e * Iv(fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(fi(t,x,y,z,vx,vy,vz))
    J = [q_e * Iv(vx * fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(vx * fi(t,x,y,z,vx,vy,vz)), q_e * Iv(vy * fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(vy * fi(t,x,y,z,vx,vy,vz)), q_e * Iv(vz * fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(vz * fi(t,x,y,z,vx,vy,vz))]

    eqs = [Dt(fe(t,x,y,z,vx,vy,vz)) ~ - Divx_v_e - Divv_F_e,
        Dt(fi(t,x,y,z,vx,vy,vz)) ~ - Divx_v_i - Divv_F_i,
        Dx(Dxϕ(t,x,y,z)) + Dy(Dyϕ(t,x,y,z)) + Dz(Dzϕ(t,x,y,z)) - μ_0*ε_0*Dt(Dtϕ(t,x,y,z)) ~ 1/ε_0*ρ,
        Dx(DxAx(t,x,y,z)) + Dy(DyAx(t,x,y,z)) + Dz(DzAx(t,x,y,z)) - μ_0*ε_0*Dt(DtAx(t,x,y,z)) ~ μ_0*J[1],
        Dx(DxAy(t,x,y,z)) + Dy(DyAy(t,x,y,z)) + Dz(DzAy(t,x,y,z)) - μ_0*ε_0*Dt(DtAy(t,x,y,z)) ~ μ_0*J[2],
        Dx(DxAz(t,x,y,z)) + Dy(DyAz(t,x,y,z)) + Dz(DzAz(t,x,y,z)) - μ_0*ε_0*Dt(DtAz(t,x,y,z)) ~ μ_0*J[3],]

    der_ = [Dx(ϕ(t,x,y,z)) ~ Dxϕ(t,x,y,z),
            Dy(ϕ(t,x,y,z)) ~ Dyϕ(t,x,y,z),
            Dz(ϕ(t,x,y,z)) ~ Dzϕ(t,x,y,z),
            Dt(ϕ(t,x,y,z)) ~ Dtϕ(t,x,y,z),
            Dx(Ax(t,x,y,z)) ~ DxAx(t,x,y,z),
            Dy(Ax(t,x,y,z)) ~ DyAx(t,x,y,z),
            Dz(Ax(t,x,y,z)) ~ DzAx(t,x,y,z),
            Dt(Ax(t,x,y,z)) ~ DtAx(t,x,y,z),
            Dx(Ay(t,x,y,z)) ~ DxAy(t,x,y,z),
            Dy(Ay(t,x,y,z)) ~ DyAy(t,x,y,z),
            Dz(Ay(t,x,y,z)) ~ DzAy(t,x,y,z),
            Dt(Ay(t,x,y,z)) ~ DtAy(t,x,y,z),
            Dx(Az(t,x,y,z)) ~ DxAz(t,x,y,z),
            Dy(Az(t,x,y,z)) ~ DyAz(t,x,y,z),
            Dz(Az(t,x,y,z)) ~ DzAz(t,x,y,z),
            Dt(Az(t,x,y,z)) ~ DtAz(t,x,y,z)]

    # Boundaries and initial conditions
    bcs_ = [fe(0,x,y,z,vx,vy,vz) ~ ic_fe(vx,vy,vz,T,m, v_drift) * geometry(x, y, z),
            fi(0,x,y,z,vx,vy,vz) ~ ic_fi(vx,vy,vz,T,m, v_drift) * geometry(x, y, z), 
            Ax(0,x,y,z) ~ q_e * Iv(vx * fe(0,x,y,z,vx,vy,vz)) + q_i * Iv(vx * fi(0,x,y,z,vx,vy,vz)) * geometry(x, y, z),
            Ay(0,x,y,z) ~ q_e * Iv(vy * fe(0,x,y,z,vx,vy,vz)) + q_i * Iv(vy * fi(0,x,y,z,vx,vy,vz)) * geometry(x, y, z),
            Az(0,x,y,z) ~ q_e * Iv(vz * fe(0,x,y,z,vx,vy,vz)) + q_i * Iv(vz * fi(0,x,y,z,vx,vy,vz)) * geometry(x, y, z),
            ϕ(0,x,y,z) ~ (q_e * Iv(fe(0,x,y,z,vx,vy,vz)) + q_i * Iv(fi(0,x,y,z,vx,vy,vz)))/ε_0 * geometry(x, y, z),
        ] 

    bcs__ = [bcs_;der_]

    # Neural Network
    CUDA.allowscalar(false)
    chain = [[FastChain(FastDense(7, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)) for _ in 1:2];
            [FastChain(FastDense(4, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)) for _ in 1:20]]
    initθ = GPU ? map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain)) : map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) 

    discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params=initθ)
    real_vars = [fe(t,x,y,z,vx,vy,vz), fi(t,x,y,z,vx,vy,vz), ϕ(t,x,y,z), Ax(t,x,y,z), Ay(t,x,y,z), Az(t,x,y,z)]
    ϕ_der_vars = [Dxϕ(t,x,y,z), Dyϕ(t,x,y,z), Dzϕ(t,x,y,z), Dtϕ(t,x,y,z)]
    Ax_der_vars = [DxAx(t,x,y,z), DyAx(t,x,y,z), DzAx(t,x,y,z), DtAx(t,x,y,z)]
    Ay_der_vars = [DxAy(t,x,y,z), DyAy(t,x,y,z), DzAy(t,x,y,z), DtAy(t,x,y,z)]
    Az_der_vars = [DxAz(t,x,y,z), DyAz(t,x,y,z), DzAz(t,x,y,z), DtAz(t,x,y,z)]
    vars = vcat(real_vars,ϕ_der_vars,Ax_der_vars,Ay_der_vars, Az_der_vars)
    @named pde_system = PDESystem(eqs, bcs__, domains, [t,x,y,z,vx,vy,vz], vars)
    prob = SciMLBase.discretize(pde_system, discretization)

    cb = function (p,l)
        println("Current loss is: $l")
        return false
    end

    # Solve
    opt = Optim.BFGS()
    res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=1000)
    phi = discretization.phi
    return phi, res
end

function save_results(phi, res, model_name)
    @save model_name*"_phi.bson" phi
    minimizers_ = [res.minimizer[s] for s in sep]
    @save model_name*"_minimizers_.bson" minimizers_
end

function load_results(phi_path, minimizers_path)
    loaded_phi = BSON.load(phi_path)[:phi]
    loaded_weights = BSON.load(minimizers_path)[:minimizers_]
end

function plot_f_3D(phi, minimizers_, model_name="")
end

function plot_E_3D(phi, minimizers_, model_name="")
end

function plot_B_3D(phi, minimizers_, model_name="")
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

function solve_electrostatic_plasma(dim=3)
    return
end

#### Zukunftsmusik


function get_initial_electrostatic_conditions(isMaxwellian=true)
    # error if f of initial condition is < 0
end

function get_initial_collisionless_conditions(isMaxwellian=true)
    # error if f of initial condition is < 0
end

function set_boundaries(type)
    # reflective
    # damped
    # periodic
end

function compose_results(dim=3)
end

function decompose_domains(dim=3)
end

#v1
# give me a domain (of type Interval(i, j))
# give me a geometry
# give me initial conditions (for f)

#>v2
# give me external coils
# give me external forces (for ICF)
# give me the properties on the boundaries
# and I give you how this plasma will move with time

# what is the strategy (the user doesn't need to know), how should it reflect in the boundaries? how many v and x dimensions? what are the bounds
# in the future we can set a geometry and size of a mesh
# it should probably start with an empty mesh, I can add the plasma geometry to the mesh, then add magnets to the mesh, and solve in the mesh which initializes all moving parts.


function solve_collisional_plasma(dim=3)
    return
end

function solve_relativistic_plasma(dim=3)
    return
end

function validate_collisionless_plasma()
    # Some method of manufactured solution or analytical approach that tells me how far off the model is
    return
end

function validate_electrostatic_plasma()
    return
end

function validate_collisional_plasma()
    return
end

function validate_relativistic_plasma()
    return
end

function plot_electrostatic_plasma(dim=3)
    return
end

function plot_collisionless_plasma(dim=3)
    return
end

