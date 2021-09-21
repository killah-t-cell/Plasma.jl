module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using NeuralPDE
import ModelingToolkit: Interval, infimum, supremum

# interface for tokamak geometry (where you can pick tokamak parameters). 
# Same for other common configurations – FRCs, Mirrors, Spheromaks... 
struct Geometry{ T }

end

#TODO learn about subtyping
struct ToroidalGeometry{ T } # <: Geometry

end
struct LinearGeometry{ T } # <: Geometry

end
# coils just really mean that at point x the magnetic field is equal to B(x) = B_plasma + B_coil. The same goes for E
struct Coils{ T } 

end
struct Walls{ T } 

end


include("model.jl")

export Geometry, ToroidalGeometry, LinearGeometry, Coils,
       solve_collisionless_plasma, solve_electrostatic_plasma, validate_collisionless_plasma, validate_electrostatic_plasma, 
       set_initial_geometry, set_space, plot_collisionless_plasma, plot_electrostatic_plasma

end