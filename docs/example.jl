using Plasma

# WHAT IT LOOKS LIKE NOW
# TODO can we somehow just let the users pick which particles they want without having to build them?
e = Species(Maxwellian(1,3,2).P)
i = Species(Maxwellian(1.1,3.3,2.3).P,1.602176634e-19, 3.3435837724e-27)

species = [e, i]
g = Geometry(v -> if (v > 0.2 && v < 0.3) 1. else 0. end)

plasma = CollisionlessPlasma(species, g)

solve(plasma)

# WHAT IT SHOULD LOOK LIKE!!
# Should geometries and distribution be structs or functions?
using Plasma
import Stellerator: Stella

sp = @species begin
    P(0.5, H₃)
    Maxwellian(0.3, H₂)
    Kappa(e)
end

g1 = ToroidalGeometry(a0=a0, R0=R0, I=2.2)

plasma1 = CollisionlessPlasma(g1, sp)
sol = solve(plasma1)

plot(sol)
sol.f([0.1,0.2,0.3,0.4,0.5,0.6])
sol.v_th([0.1,0.2,0.3,0.4,0.5,0.6])

# Ex 2
g2 = Geometry(Stella) # Bitvector with Stellerator format
plasma2 = Electrostatic(g2, sp)
sol = solve(plasma2, dim=3)

plot(sol.n)
plot(sol.ω_ce)


function explode(arg::T) where {T <: Number}
end

