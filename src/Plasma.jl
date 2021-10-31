module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using NeuralPDE
import ModelingToolkit: Interval, infimum, supremum

# interface for tokamak geometry (where you can pick tokamak parameters). 
# Same for other common configurations – FRCs, Mirrors, Spheromaks... 


# distribution.jl, plasma.jl (define coil, plasma, and geometry here), solve.jl, analyze.jl

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
    temperature::Float64 # eV – compute from distributions
    density::Float64 # g/cc – compute from distributions
    species::Vector{Species} # force same length as initial distributions
    initial_distributions::Vector{Distribution}
    geometry::Geometry
end

# abstract type PlasmaGeometry <: Geometry end

# abstract type ToroidalPlasmaGeometry <: PlasmaGeometry end


include("model.jl")

# export TODO

end