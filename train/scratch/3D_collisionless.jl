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
    @variables fe(..) fi(..) ϕ(..) Ax(..) Ay(..) Az(..)
    @variables Dxϕ(..) Dyϕ(..) Dzϕ(..) Dtϕ(..) DxAx(..) DyAx(..) DzAx(..) DtAx(..) DxAy(..) DyAy(..) DzAy(..) DtAy(..) DxAz(..) DyAz(..) DzAz(..) DtAz(..)
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    Dvx = Differential(vx)
    Dvy = Differential(vy)
    Dvz = Differential(vz)
    Dt = Differential(t)

    # Registrations # TODO – how to pass functions and register
    geometry(x,y,z) = geometry_(x,y,z)
    ic_fe(vx,vy,vz,T,m, v_drift) = ic_fe_(vx,vy,vz,T,m, v_drift)
    ic_fi(vx,vy,vz,T,m, v_drift) = ic_fi_(vx,vy,vz,T,m, v_drift)
    @register geometry(x,y,z)
    @register ic_fe(vx,vy,vz,T,m, v_drift)
    @register ic_fi(vx,vy,vz,T,m, v_drift)

    # Constants
    μ_0 = 1.25663706212e-6 # N A⁻²
    ε_0 = 8.8541878128e-12 # F ms⁻¹
    q_e   = 1.602176634e-19 # Coulombs
    q_i   = 1.602176634e-19 # Coulombs #TODO this change should be Z, and Z should be a parameter
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
    Div_B = Dx(B[1]) + Dy(B[2]) + Dz(B[3])
    Div_E = Dx(E[1]) + Dy(E[2]) + Dz(E[3])

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
            Div_B ~ 0,
            Div_E ~ (q_e * Iv(fe(0,x,y,z,vx,vy,vz)) + q_i * Iv(fi(0,x,y,z,vx,vy,vz)))/ε_0 * geometry(x, y, z),
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





#### with E and B

@parameters t x y z vx vy vz
@variables fe(..) fi(..) Ex(..) Ey(..) Ez(..) Bx(..) By(..) Bz(..)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)
Dvx = Differential(vx)
Dvy = Differential(vy)
Dvz = Differential(vz)
Dt = Differential(t)

# Constants
μ_0 = 1.25663706212e-6 # N A⁻²
ε_0 = 8.8541878128e-12 # F ms⁻¹
q_e   = 1.602176634e-19 # Coulombs
q_i   = 1.602176634e-19 # Coulombs #TODO this change should be Z, and Z should be a parameter
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



function solve_collisionless_plasma(geometry_, lb, ub, T, ic_fe_, ic_fi_=ic_fe_, v_drift=zeros(3), time_lb=lb, time_ub=ub, GPU=true)
        if lb > ub
            error("lower bound must be larger than upper bound")
        end
    
        if ic_fe(args...) < 0 || ic_fi(args...) < 0
            error("distribution function must be greater than 0")
        end
    
        # TODO check if it is in the collisionless regime
    
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
    
        # Registrations # TODO – how to pass functions and register
        geometry(x,y,z) = geometry_(x,y,z)
        ic_fe(vx,vy,vz,T,m, v_drift) = ic_fe_(vx,vy,vz,T,m, v_drift)
        ic_fi(vx,vy,vz,T,m, v_drift) = ic_fi_(vx,vy,vz,T,m, v_drift)
        @register geometry(x,y,z)
        @register ic_fe(vx,vy,vz,T,m, v_drift)
        @register ic_fi(vx,vy,vz,T,m, v_drift)
    
        # Constants
        μ_0 = 1.25663706212e-6 # N A⁻²
        ε_0 = 8.8541878128e-12 # F ms⁻¹
        q_e   = 1.602176634e-19 # Coulombs
        q_i   = 1.602176634e-19 # Coulombs #TODO this change should be Z, and Z should be a parameter
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
        Div_B = Dx(B[1]) + Dy(B[2]) + Dz(B[3])
        Div_E = Dx(E[1]) + Dy(E[2]) + Dz(E[3])
    
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
                Div_B ~ 0,
                Div_E ~ (q_e * Iv(fe(0,x,y,z,vx,vy,vz)) + q_i * Iv(fi(0,x,y,z,vx,vy,vz)))/ε_0 * geometry(x, y, z),
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



    function solve_collisionless_plasma_with_E_B(geometry_, lb, ub, T, ic_fe_, ic_fi_=ic_fe_, v_drift=zeros(3), time_lb=lb, time_ub=ub, GPU=true)
        if lb > ub
            error("lower bound must be larger than upper bound")
        end
    
        if ic_fe(args...) < 0 || ic_fi(args...) < 0
            error("distribution function must be greater than 0")
        end
    
        # TODO check if it is in the collisionless regime
    
        @parameters t x y z vx vy vz
        @variables fe(..) fi(..) Ex(..) Ey(..) Ez(..) Bx(..) By(..) Bz(..)
        Dx = Differential(x)
        Dy = Differential(y)
        Dz = Differential(z)
        Dvx = Differential(vx)
        Dvy = Differential(vy)
        Dvz = Differential(vz)
        Dt = Differential(t)
        
        # Constants
        μ_0 = 1.25663706212e-6 # N A⁻²
        ε_0 = 8.8541878128e-12 # F ms⁻¹
        q_e   = 1.602176634e-19 # Coulombs
        q_i   = 1.602176634e-19 # Coulombs #TODO this change should be Z, and Z should be a parameter
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