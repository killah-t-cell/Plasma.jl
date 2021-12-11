module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using GLMakie
using CUDA
using NeuralPDE
using LinearAlgebra
using DomainSets
import ModelingToolkit: Interval, infimum, supremum

abstract type AbstractPlasma end

abstract type AbstractGeometry end

abstract type AbstractDistribution end

abstract type AbstractCoil end

"""
Species holds

q – charge in C
m – mass in Kg

to describe a particle species
"""
struct Species{ T <: Number }
    q::T # charge in C
    m::T # mass in Kg

    function Species(q=-1.602176634e-19, m=9.10938188e-31)
        new{typeof(q)}(
            q, m
        )
    end
end

"""
Velocity distribution of particles

P – probability function
species – which species of particle
"""
struct Distribution <: AbstractDistribution
    P::Function
    species::Species
end

"""
Describes the initial geometry of a plasma

f – conditional function

# Example

Geometry(x -> x > 0.4 ? 1. : 0)
"""
struct Geometry{F <: Function} <: AbstractGeometry
    f::F

    function Geometry(f= _ -> 1)
        new{typeof(f)}(f)
    end
end

"""
CollisionlessPlasma object that can be passed to Plasma.solve for simulation

It takes a geometry and a vector of distributions (one for every particle).
"""
struct CollisionlessPlasma{ G <: AbstractGeometry } <: AbstractPlasma
    distributions::Vector{Distribution}
    geometry::G
end

"""
ElectrostaticPlasma object that can be passed to Plasma.solve for simulation

It takes a geometry and a vector of distributions (one for every particle).
"""
struct ElectrostaticPlasma{ G <: AbstractGeometry } <: AbstractPlasma
    distributions::Vector{Distribution}
    geometry::G
end

"""
Object to hold (and save) the results of Plasma.solve.

It consists of

plasma – CollisionlessPlasma or ElectrostaticPlasma object
vars – dependent variables
dict_vars – dictionary of dependent variables
phi – trained trial solution that approximates plasma movement
res – result of optimization
initθ – weights of the neural network
domains – domains of the simulation
"""
struct PlasmaSolution{ P <: AbstractPlasma, V, DV, PHI, RE, IN, DO }
    plasma::P
    vars::V
    dict_vars::DV
    phi::PHI
    res::RE
    initθ::IN
    domains::DO
end
struct Constants{ T <: Number}
    μ_0::T
    ϵ_0::T

    function Constants()
        μ_0 = 1.25663706212e-6 # N A⁻²
        ϵ_0 = 8.8541878128e-12 # F ms⁻¹
        new{typeof(μ_0)}(μ_0, ϵ_0)
    end
end

const species = (
    e = Species(),
    p = Species(1.602176634e-19 ,1.673523647e-27),
    D = Species(1.602176634e-19 ,3.344476425e-27),
    T = Species(1.602176634e-19 ,5.00735588e-27),
    He³ = Species(1.602176634e-19 ,5.00641192e-27),
    He⁴ = Species(1.602176634e-19 ,6.64465620e-27),
)

include("distribution.jl")
include("boundaries.jl")
include("solve.jl")
include("geometry.jl")
include("analyze.jl")

export CollisionlessPlasma, ElectrostaticPlasma, PlasmaSolution
export Distribution, Maxwellian
export Neumann, Dirichlet, Reflective
export Geometry
export Species, species
export Constants

end