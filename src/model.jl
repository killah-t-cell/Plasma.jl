using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
using DomainSets
import ModelingToolkit: Interval, infimum, supremum
using Parameters
using Plots
using CUDA
using BSON: @save
using BSON: @load
@with_kw struct PlasmaParameters{T, F}
    geometry::F 
    T::Float64 # Kelvin
    IC_e::F
    IC_i::F
    v_drift::Vector{T} = zeros(3)
    m_i::Float64 = 3.3435837724e-27
end

# Boundaries
function set_boundaries(type)
    # reflective
    # damped
    # periodic
end

# Distribution functions
"""
Maxwellian velocity distribution in 3 dimensions
"""
function maxwellian_3D(vx,vy,vz,T,m, v_drift)
    Kb = 1.3806503e-23
    v_th = sqrt(2*Kb*T/m)
    v = sqrt(vx^2 + vy^2 + vz^2)
    v_drift = sqrt(v_drift[1]^2 + v_drift[2]^2 + v_drift[3]^2)
    return (π*v_th^2)^(-3/2) * exp(-(v - v_drift)^2/v_th^2)
end

"""
Maxwellian velocity distribution in 1 dimension
"""
function maxwellian_1D(v,T,v_th, v_drift)
    Kb = 1.3806503e-23
    v_th = sqrt(2*Kb*T/m)
    return (π*v_th^2)^(-3/2) * exp(-(v - v_drift)^2/v_th^2)
end

# Models
"""
Solves a collisionless plasma with 2 species in 6 dimensions

Collisionless kinetic plasmas are accurate models when the plasma is at high temperature, is of low density, and that collisions are unimportant.

High temperature means -> 

For more information: https://doi.org/10.1137/1.9781611971477
"""
function solve_collisionless_plasma(params, lb, ub, time_lb=lb, time_ub=ub, GPU=true)
    if lb > ub
        error("lower bound must be larger than upper bound")
    end

    if ic_fe(args...) < 0 || ic_fi(args...) < 0
        error("distribution function must be greater than 0")
    end

    @parameters t x y z vx vy vz
    @variables fe(..) fi(..) Ex(..) Ey(..) Ez(..) Bx(..) By(..) Bz(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    Dvx = Differential(vx)
    Dvy = Differential(vy)
    Dvz = Differential(vz)
    Dt = Differential(t)

    # Registrations
    v_drift = params.v_drift
    @register params.geometry(x,y,z)
    @register params.IC_e(vx,vy,vz,T,m, v_drift)
    @register params.IC_i(vx,vy,vz,T,m, v_drift)

    # Constants
    μ_0 = 1.25663706212e-6 # N A⁻²
    ε_0 = 8.8541878128e-12 # F ms⁻¹
    q_e   = 1.602176634e-19 # Coulombs
    q_i   = 1.602176634e-19 # Coulombs
    m_e = 9.10938188e-31 # Kg
    m_i = 3.3435837724e-27 # Deuterium mass

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
    E = [Ex(t,x,y,z), Ey(t,x,y,z), Ez(t,x,y,z)]
    B = [Bx(t,x,y,z), By(t,x,y,z), Bz(t,x,y,z)]
    
    v = [vx, vy, vz]
    Divx_v_e = vx * Dx(fe(t,x,y,z,vx,vy,vz)) + vy * Dy(fe(t,x,y,z,vx,vy,vz)) + vz * Dz(fe(t,x,y,z,vx,vy,vz))
    Divx_v_i = vx * Dx(fi(t,x,y,z,vx,vy,vz)) + vy * Dy(fi(t,x,y,z,vx,vy,vz)) + vz * Dz(fi(t,x,y,z,vx,vy,vz))
    F_e = q_e/m_e * (E + cross(v, B))
    F_i = q_i/m_i * (E + cross(v, B))
    DfDv_e = [Dvx(fe(t,x,y,z,vx,vy,vz)), Dvy(fe(t,x,y,z,vx,vy,vz)), Dvz(fe(t,x,y,z,vx,vy,vz))]
    DfDv_i = [Dvx(fi(t,x,y,z,vx,vy,vz)), Dvy(fi(t,x,y,z,vx,vy,vz)), Dvz(fi(t,x,y,z,vx,vy,vz))]
    Divv_F_e = dot(F_e, DfDv_e)
    Divv_F_i = dot(F_i, DfDv_i)
    Div_B = Dx(B[1]) + Dy(B[2]) + Dz(B[3])
    Div_E = Dx(E[1]) + Dy(E[2]) + Dz(E[3])
    
    ρ = q_e * Iv(fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(fi(t,x,y,z,vx,vy,vz))
    J = [q_e * Iv(vx * fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(vx * fi(t,x,y,z,vx,vy,vz)), q_e * Iv(vy * fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(vy * fi(t,x,y,z,vx,vy,vz)), q_e * Iv(vz * fe(t,x,y,z,vx,vy,vz)) + q_i * Iv(vz * fi(t,x,y,z,vx,vy,vz))]
    
    eqs = [Dt(fe(t,x,y,z,vx,vy,vz)) ~ - Divx_v_e - Divv_F_e,
           Dt(fi(t,x,y,z,vx,vy,vz)) ~ - Divx_v_i - Divv_F_i,
           curl(E)[1] ~ Dt(Bx(t,x,y,z)),
           curl(E)[2] ~ Dt(By(t,x,y,z)),
           curl(E)[3] ~ Dt(Bz(t,x,y,z)),
           ε_0*μ_0 * Dx(E[1]) - curl(B)[1] ~ - μ_0*J[1],
           ε_0*μ_0 * Dx(E[2]) - curl(B)[2] ~ - μ_0*J[2],
           ε_0*μ_0 * Dx(E[3]) - curl(B)[3] ~ - μ_0*J[3],
           Div_E ~ ρ/ε_0,
           Div_B ~ 0]
    
    # Boundaries and initial conditions
    bcs_ = [fe(0,x,y,z,vx,vy,vz) ~ ic_fe(vx,vy,vz,T,m, v_drift) * geometry(x, y, z),
            fi(0,x,y,z,vx,vy,vz) ~ ic_fi(vx,vy,vz,T,m, v_drift) * geometry(x, y, z), 
            Div_B ~ 0,
            Div_E ~ (q_e * Iv(fe(0,x,y,z,vx,vy,vz)) + q_i * Iv(fi(0,x,y,z,vx,vy,vz)))/ε_0 * geometry(x, y, z)] 
    
    bcs__ = [bcs_;der_]
    
    # Neural Network
    CUDA.allowscalar(false)
    chain = [[FastChain(FastDense(7, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)) for _ in 1:2];
            [FastChain(FastDense(4, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)) for _ in 1:20]]
    initθ = GPU ? map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain)) : map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) 
    
    discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params=initθ)
    vars = [fe(t,x,y,z,vx,vy,vz), fi(t,x,y,z,vx,vy,vz), Ex(t,x,y,z), Ey(t,x,y,z), Ez(t,x,y,z), Bx(t,x,y,z), By(t,x,y,z), Bz(t,x,y,z)]
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

# Save & load methods
function save_results(phi, res, model_name)
    @save model_name*"_phi.bson" phi
    minimizers_ = [res.minimizer[s] for s in sep]
    @save model_name*"_minimizers_.bson" minimizers_
end

function load_results(phi_path, minimizers_path)
    loaded_phi = BSON.load(phi_path)[:phi]
    loaded_weights = BSON.load(minimizers_path)[:minimizers_]
    return loaded_phi, loaded_weights
end

# Plotting functions
# TODO maybe unite these function so one function plots f, E, and B
function plot_3D(phi, minimizers_, model_name="")
    # plots f, E, B    
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


#= Zukunftsmusik


function get_initial_electrostatic_conditions(isMaxwellian=true)
    # error if f of initial condition is < 0
end

function get_initial_collisionless_conditions(isMaxwellian=true)
    # error if f of initial condition is < 0
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

=#