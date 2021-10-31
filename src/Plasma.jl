module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using NeuralPDE
import ModelingToolkit: Interval, infimum, supremum

# interface for tokamak geometry (where you can pick tokamak parameters). 
# Same for other common configurations – FRCs, Mirrors, Spheromaks... 


# distribution.jl, plasma.jl, solve.jl, analyze.jl

# abstract type AbstractObject end # so we can specify a wall that logs when a plasma hits it.

abstract type AbstractCoil end
 
abstract type AbstractPlasma end

abstract type AbstractGeometry end

abstract type AbstractDistribution end

struct MaxwellianDistribution <: AbstractDistribution
end


struct Species
    q::Float64
    m::Float64
    concentration::Float16 # defaults to 1
end

# given f(t,x,v) we can compute density, temperature, and all other parameters. It is probably wise to store T, and n, and use that as inputs to functions that compute all parameters.
struct CollisionlessPlasma <: AbstractPlasma
    temperature::Float64 # eV
    density::Float64 # g/cc
    species::Vector{Species}
    initial_distributions::Vector{Distribution}
    geometry::Geometry
end

# abstract type PlasmaGeometry <: Geometry end

# abstract type ToroidalPlasmaGeometry <: PlasmaGeometry end





#TODO learn about subtyping
struct ToroidalGeometry{ T } # <: Geometry

end
struct LinearGeometry{ T } # <: Geometry

end
# coils just really mean that at point x the magnetic field is equal to B(x) = B_plasma + B_coil. The same goes for E
struct Coils{ T } # <: Object

end
struct Walls{ T } # <: Object

end


include("model.jl")

export Geometry, ToroidalGeometry, LinearGeometry, Coils,
       solve_collisionless_plasma, solve_electrostatic_plasma, validate_collisionless_plasma, validate_electrostatic_plasma, 
       set_initial_geometry, set_space, plot_collisionless_plasma, plot_electrostatic_plasma

end