using Plasma

e = Species(Maxwellian(1,3,2).P)
i = Species(Maxwellian(1.1,3.3,2.3).P,1.602176634e-19, 3.3435837724e-27)

species = [e, i]
g = Geometry(v -> if (v > 0.2 && v < 0.3) 1. else 0. end)

g isa AbstractGeometry
plasma = CollisionlessPlasma([e, i], g)