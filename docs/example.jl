using Plasma

# WHAT IT LOOKS LIKE NOW
# TODO can we somehow just let the users pick which particles they want without having to build them?
e = Species(Maxwellian(1,3,2).P)
i = Species(Maxwellian(1.1,3.3,2.3).P,1.602176634e-19, 3.3435837724e-27)

species = [e]
g = Geometry()

plasma = ElectrostaticPlasma(species, g)

Plasma.solve(plasma, g)

# WHAT IT SHOULD LOOK LIKE!!
# Should geometries and distribution be structs or functions?
# How do I define some useful atoms? Use the Species struct for it. https://github.com/JuliaPhysics/PeriodicTable.jl/blob/master/src/PeriodicTable.jl # const elements = Elements(_elements_data)
# start with e, p, H₂, H₃, He₄, Li₆, Li₇, B₁₁
# species (IC, m, q), number of species, geometry, plasma type.

using Plasma

sp = [P(species.H₃), Maxwellian(T, v, species.H₂.m), Kappa(T,v, species.e.m)]

    species[H₃]

g = ToroidalGeometry(a0=a0, R0=R0, I=2.2)

plasma = CollisionlessPlasma(g, sp)
sol = solve(plasma) 

plot(sol)

point = [0.1,2.2,9.3,7.4,4.8,3.6]
sol.f(point)
sol.v_th(point .+ 0.1)


#=
# Ex 2
import Stellerator: Stella

g2 = Geometry(Stella) # Bitvector with Stellerator format
plasma2 = Electrostatic(g2, sp)
sol = solve(plasma2, dim=3)

plot(sol.n)
plot(sol.ω_ce)

sp = @species begin
    H₃ -> P, 0.5
    H₂ -> Maxwellian, 0.5
    e -> Kappa, 0.5
end
=#

using Plasma

@species e H₃
geometry = ToroidalGeometry(a0=a0,r0=r0)
