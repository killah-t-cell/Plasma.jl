# what is the strategy (the user doesn't need to know), how should it reflect in the boundaries? how many v and x dimensions? what are the bounds
# in the future we can set a geometry and size of a mesh
# it should probably start with an empty mesh, I can add the plasma geometry to the mesh, then add magnets to the mesh, and solve in the mesh which initializes all moving parts.
function solve_collisionless_plasma(initial_condition::Vector{Equation}, boundary_type; v_dim=3, x_dim=3, lower_bound=0, upper_bound=1)
    # if f of initial condition is < 0 or lower bound is higher than upper bound
    # throw error with message

    # Dependent and independent variables
    
    # Constants
   
    # Integrals

    # Helpers

    # Equations


    # Boundary conditions

    # Domains

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




