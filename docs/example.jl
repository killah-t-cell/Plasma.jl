using Plasma

TD = 30000 # eV
Te = 10000 # eV

D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D) 
D_e = Distribution(Maxwellian(Te, e.m), e) 
G = Geometry() 

plasma = ElectrostaticPlasma([D_D, D_e], G)

# example setting custom initial and boundary conditions
Es, xs, vs, t = Plasma.get_vars(plasma; dim=1)
ic = [Es[1](0, xs...) ~ cos(xs[1])]
bc = [Es[1](t,0.4) ~ cos(t)]

sol = Plasma.solve(plasma, dim=1, GPU=false, conditions = [ic;bc]) 

Plasma.plot(sol)

## 2D with custom P and species
using Plasma

Tα = 70000 # eV

α = Species(1.602176634e-19, 6.6446562e-27)

function HotCarrier(T) 
    Kb = 8.617333262145e-5
    P(x,v) = exp(-v/(Kb*T))
end

Dα = Distribution(HotCarrier(Tα), α)
G = Geometry() # TODO define a custom geometry

plasma = ElectrostaticPlasma([Dα], G)

Plasma.solve(plasma, dim=2) # with GPU

Plasma.plot(sol)


## 3D CollisionlessPlasma
using Plasma

TD = 15000 # eV
TT = 15000 # eV
Te = 13000 # eV

e = species.e
T = species.T
D = species.D

De = Distribution(Maxwellian(Te, e.m), e)
DT = Distribution(Maxwellian(TT, T.m), T)
DD = Distribution(Maxwellian(TD, D.m), D)
G = Geometry()

plasma = CollisionlessPlasma([De,DT,DD], G)

Plasma.solve(plasma)

Plasma.plot(sol)
