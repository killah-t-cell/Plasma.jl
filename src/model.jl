using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

# (Vlasov)
# 1D plasma solver
# 2D plasma solver
# 3D plasma solver
# Electrostatic solver in 1D, 2D, 3D (Vlasov Poisson)


# decompose the domain whenever I solve anything
# Default to retrain when I use the same equation on a different domain and see if it does the job

 
@parameters t x y z
@variables u1(..) u2(..) u3(..) w(..)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)
Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dzz = Differential(z)^2
Dtt = Differential(t)^2

# Constants & non homogeneous terms
a = 5
ρ(t, x, y, z) = exp(-t^2) + 3 * y + z + x^2
J1(t, x, y, z) = exp(-t^2) + 3 * y + z + x^2
J2(t, x, y, z) = 0
J3(t, x, y, z) = 0

# Analytic solution
w_homogeneous_analytic(t, x, y, z) = sin(x + 1) * sin(y + 1) * sin(z + 1) * sin(a * t * sqrt(3))
dw_homogeneous_analytic(t, x, y, z) = sqrt(3) + a * sin(x + 1) * sin(y + 1) * sin(z + 1) * cos(a * t * sqrt(3))


# Vector calculus
laplacian_w  = Dxx(w(t, x, y, z))  + Dyy(w(t, x, y, z))  + Dzz(w(t, x, y, z))
laplacian_u1 = Dxx(u1(t, x, y, z)) + Dyy(u1(t, x, y, z)) + Dzz(u1(t, x, y, z))
laplacian_u2 = Dxx(u2(t, x, y, z)) + Dyy(u2(t, x, y, z)) + Dzz(u2(t, x, y, z))
laplacian_u3 = Dxx(u3(t, x, y, z)) + Dyy(u3(t, x, y, z)) + Dzz(u3(t, x, y, z))
divergence_u = Dx(u1(t, x, y, z)) + Dy(u2(t, x, y, z)) + Dz(u3(t, x, y, z))

# Lorenz gauge system of nonhomogeneous wave equations
eqs = [Dtt(w(t, x, y, z)) ~ a^2 * laplacian_w + ρ(t, x, y, z),
       Dtt(u1(t, x, y, z)) ~ a^2 * laplacian_u1 + J1(t, x, y, z),
       Dtt(u2(t, x, y, z)) ~ a^2 * laplacian_u2 + J2(t, x, y, z),
       Dtt(u3(t, x, y, z)) ~ a^2 * laplacian_u3 + J3(t, x, y, z),
       a^2 * Dt(w(t, x, y, z)) ~ divergence_u]


# 1D Vlasov
@parameters t x v
@variables fi(..) fe(..) V(..) A1(..) A2(..) A3(..) E1(..) E2(..) E3(..) B1(..) B2(..) B3(..)
Dxx = Differential(x)^2
Dx = Differential(x)
Dtt = Differential(t)^2
Dt = Differential(t)
Dv = Differential(v)

# Constants
μ_0 = 1.25663706212e-6 # N A⁻²
ε_0 = 8.8541878128e-12 # F ms⁻¹
q   = 1.602176634e-19 # Coulombs
m   = 3.34358377241e-27 # Kg

# Integrals
Ix = Integral(x, ClosedInterval(0, 1)) 
Iy = Integral(y, ClosedInterval(0, 1))
Iz = Integral(z, ClosedInterval(0, 1)) 

# Helpers
divergence(a)  = Dx(a[1]) + Dy(a[2]) + Dz(a[3])
gradV = [Dx(V(t, x, y, z)), Dy(V(t, x, y, z)), Dz(V(t, x, y, z))]
LaplacianV = Dxx(V(t, x, y, z)) + Dyy(V(t, x, y, z)) + Dzz(V(t, x, y, z))

# what is the strategy (the user doesn't need to know), how should it reflect in the boundaries? how many v and x dimensions? what are the bounds
function solve_collisionless_plasma(initial_condition; v_dim=3, x_dim=3, lower_bound=0, upper_bound=1)
    # if f of initial condition is < 0 or lower bound is higher than upper bound
    # throw error with message

    # Dependent and independent variables
    @parameters t, x, y, z, vx, vy, vz
    @variables f(..), V(..)
    Dxx = Differential(x)^2
    Dyy = Differential(y)^2
    Dzz = Differential(z)^2
    Dtt = Differential(t)^2
    Dx = Differential(x)
    Dy = Differential(y)
    Dz = Differential(z)
    Dt = Differential(t)

    # Constants
    μ_0 = 1.25663706212e-6 # N A⁻²
    ε_0 = 8.8541878128e-12 # F ms⁻¹
    q   = 1.602176634e-19 # Coulombs
    m   = 3.34358377241e-27 # Kg

    # Integrals
    Ix = Integral(x, ClosedInterval(0, 1)) 
    Iy = Integral(y, ClosedInterval(0, 1))
    Iz = Integral(z, ClosedInterval(0, 1)) 

    # Helpers
    divergence(a)  = Dx(a[1]) + Dy(a[2]) + Dz(a[3])
    gradV = [Dx(V(t, x, y, z)), Dy(V(t, x, y, z)), Dz(V(t, x, y, z))]
    LaplacianV = Dxx(V(t, x, y, z)) + Dyy(V(t, x, y, z)) + Dzz(V(t, x, y, z))

    # Equations


    # Boundary conditions

    # Domains

    # Neural network


    # return discretization.phi, res



function solve_electrostatic_plasma(dim=3)
    return
end

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




