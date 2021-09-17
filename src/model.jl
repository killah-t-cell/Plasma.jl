



function set_initial_geometry()
end

function get_initial_electrostatic_conditions(isMaxwellian=true)
    # error if f of initial condition is < 0
end

function get_initial_collisionless_conditions(isMaxwellian=true)
    # error if f of initial condition is < 0
end

function set_space(dim=3)
    # error if lower bound is higher than upper bound
    # return domain, t_min, t_max
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

# give me a domain (of type Interval(i, j))
# give me a geometry
# give me external coils
# give me external forces (for ICF)
# give me initial conditions
# give me the properties on the boundaries
# and I give you how this plasma will move with time

# what is the strategy (the user doesn't need to know), how should it reflect in the boundaries? how many v and x dimensions? what are the bounds
# in the future we can set a geometry and size of a mesh
# it should probably start with an empty mesh, I can add the plasma geometry to the mesh, then add magnets to the mesh, and solve in the mesh which initializes all moving parts.
function solve_collisionless_plasma()
    # if lower bound is higher than upper bound
    # throw error with message

    # Dependent and independent variables
    
    # Domains

    # Constants
   
    # Integrals

    # Helpers

    # Equations

    # Geometry

    # Boundary conditions

    # Neural network

    # return discretization.phi, res
    return
end


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

function plot_electrostatic_plasma(dim=3)
    return
end

function plot_collisionless_plasma(dim=3)
    return
end

cb = function (p,l)
    println("Current loss is: $l")
    return false
end