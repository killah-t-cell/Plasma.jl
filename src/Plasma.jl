module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using NeuralPDE
using Parameters
import ModelingToolkit: Interval, infimum, supremum

# given f(t,x,v) we can compute density, temperature, and all other parameters. It is probably wise to store T, and n, and use that as inputs to functions that compute all parameters.

# TODO abstract type AbstractObject end # so we can specify a wall that logs when a plasma hits it.

# Abstracts
abstract type AbstractCoil end
 
abstract type AbstractPlasma end

abstract type AbstractGeometry end

abstract type AbstractDistribution end

# Geometries
# TODO add ToroidalGeometry, LinearGeometry and other structs that let you pick parameters (like Toroidal parameters to define a plasma)
@with_kw struct Geometry{F <: Function} <: AbstractGeometry
    f::F = (_) -> 1 # defining function
end

# Distributions
@with_kw struct Maxwellian{Ty, V} <: AbstractDistribution
    v::V
    v_drift::V = zeros(length(v))
    T::Ty # Temperature in eV
    m::Ty # Mass in Kg
    P::Ty = nothing # probability distribution : 0 ≤ P ≤ 1

    function Maxwellian(v, T, m; P=nothing, v_drift=zeros(length(v)))
        if !(v isa Array)
            v = [v]    
        end

        if length(v) != length(v_drift)
            error("v and v_drift should have the same length")
        end

        v_ = sqrt(sum(v .^2))
        v_drift_ = sqrt(sum(v_drift.^2))
        Kb = 8.617333262145e-5
        v_th = sqrt(2*Kb*T/m)
        P = (π*v_th^2)^(-3/2) * exp(-(v_ - v_drift_)^2/v_th^2)

        new{typeof(m),typeof(v)}(
            v, v_drift, T, m, P
        )

    end
end

# Species
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

# Plasma
struct CollisionlessPlasma{ T, G <: AbstractGeometry } <: AbstractPlasma
    species::Vector{Species{T}}
    geometry::G
end


include("model.jl")

# export TODO

end