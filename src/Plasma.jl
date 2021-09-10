module Plasma

using Flux
using ModelingToolkit
using GalacticOptim
using DiffEqFlux
using NeuralPDE
import ModelingToolkit: Interval, infimum, supremum

include("model.jl")
#=
# give me a domain (of type Interval(i, j))
# give me a geometry
# give me external coils
# give me external forces (for ICF)
# give me initial conditions
# give me the properties on the boundaries
# and I give you how this plasma will move with time
maxwellian = 1

# set a box of n dimensions with a shape of the plasma with coils in certain places
# coils just really mean that at point x the magnetic field is equal to B(x) = B_plasma + B_coil. The same goes for E
function set_geometry()
    return
end

# interface for tokamak geometry (where you can pick tokamak parameters). 
# Same for other common configurations – FRCs, Mirrors, Spheromaks... get contributors to do this

function set_external_forces()
    return
end

function get_u0(domain, temperature, geometry=domain, distribution=maxwellian)
    return
end

function model_plasma(domain, u0; geometry=domain, coils=nothing, dim=3)
    return
end
=#
end