# given f(t,x,v) we can compute density, temperature, and all other parameters. It is probably wise to store T, and n, and use that as inputs to functions that compute all parameters.

# TODO maybe plasma should have a phi, and weights field to save the output with the input? 
#Or maybe make a PlasmaSolution struct that stores the plasma and the rest (including lb, ub)
#=
function plot(plasma::CollisionlessPlasma)
end

function plot(plasma::ElectrostaticPlasma)
end
=#