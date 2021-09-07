using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum


# what is the strategy (the user doesn't need to know), how should it reflect in the boundaries? how many v and x dimensions? what are the bounds
# in the future we can set a geometry and size of a mesh
# it should probably start with an empty mesh, I can add the plasma geometry to the mesh, then add magnets to the mesh, and solve in the mesh which initializes all moving parts.
function solve_collisionless_plasma(initial_condition::Vector{Equation}, boundary_type; v_dim=3, x_dim=3, lower_bound=0, upper_bound=1)
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




