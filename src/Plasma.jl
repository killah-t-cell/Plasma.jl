module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using NeuralPDE
using Parameters
import ModelingToolkit: Interval, infimum, supremum

include("solve.jl")
include("geometry.jl")
include("distribution.jl")
include("analyze.jl")

abstract type AbstractPlasma end
struct Species{ T <: Number }
    q::T # charge in C
    m::T # mass in Kg
    P::T # probability distribution : 0 ≤ P ≤ 1

    function Species(P, q=1.602176634e-19, m=9.10938188e-31)
        if P < 0.0
            error("distribution should be non-negative")
        end
        if P > 1.0
            error("distribution should not exceed 1")
        end
        new{typeof(q)}(
            q, m, P
        )
    end
end
struct CollisionlessPlasma{ T, G <: AbstractGeometry } <: AbstractPlasma
    species::Vector{Species{T}}
    geometry::G
end
struct ElectrostaticPlasma{ T, G <: AbstractGeometry } <: AbstractPlasma
    species::Vector{Species{T}}
    geometry::G
end

export CollisionlessPlasma, ElectrostaticPlasma
export AbstractDistribution, Maxwellian
export Geometry
export Species

end