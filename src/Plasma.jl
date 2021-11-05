module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using NeuralPDE
using LinearAlgebra
using Parameters
import ModelingToolkit: Interval, infimum, supremum

include("solve.jl")
include("geometry.jl")
include("distribution.jl")
include("analyze.jl")

abstract type AbstractPlasma end

# TODO Species could be more elegant. It feels a bit hacky
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
struct Constants{ T <: Number}
    μ_0::T
    ϵ_0::T

    function Constants()
        μ_0 = 1.25663706212e-6 # N A⁻²
        ϵ_0 = 8.8541878128e-12 # F ms⁻¹
        new{typeof(μ_0)}(μ_0, ϵ_0)
    end
end

export CollisionlessPlasma, ElectrostaticPlasma
export AbstractDistribution, Maxwellian
export Geometry
export Species
export Constants

end