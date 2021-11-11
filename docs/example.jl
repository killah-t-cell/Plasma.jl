using Plasma

TD = 30000 # eV
Te = 10000 # eV

D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D) 
D_e = Distribution(Maxwellian(Te, e.m), e) 
G = Geometry() 

plasma = ElectrostaticPlasma([D_D, D_e], G)

sol = Plasma.solve(plasma, dim=1, GPU=false) 

Plasma.plot(sol)

## 2D with custom P and species
using Plasma

Tα = 70000 # eV

α = Species(1.602176634e-19, 6.6446562e-27)

function Kappa(T, m) 
    P(x,v) = sqrt(sum(v .^2)) * x + m / exp(T)
end

Dα = Distribution(Kappa(Tα, α.m), α)
G = Geometry() # TODO define a custom geometry

plasma = ElectrostaticPlasma([Dα], G)

Plasma.solve(plasma, dim=2) # with GPU

plot(sol)
# TODO check the value of n and T and other plasma parameter at point T


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

plot(sol)


using Plasma

D = species.D
e = species.e

D_D = Distribution(Maxwellian(3e5, D.m), D) 
D_e = Distribution(Maxwellian(1.2e5, e.m), e) 
G = Geometry()

plasma = ElectrostaticPlasma([D_D, D_e], G)

sol = Plasma.solve(plasma) 
plot(sol)