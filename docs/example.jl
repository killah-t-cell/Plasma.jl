#### Pretty example program
using Plasma

TD = 30000 # eV
Te = 10000 # eV

D = species.D
e = species.e
e.q
e.m

D_D = Distribution(Maxwellian(TD, D.m), D)
D_e = Distribution(Maxwellian(Te, e.m), e)
G = Geometry()

typeof([D_D, D_e])
plasma = ElectrostaticPlasma([D_D, D_e], G)

Plasma.solve(plasma, dim=1, GPU=false) 


#### custom P example 
using Plasma

TD = 30000 # eV
Te = 10000 # eV

D = species.D
e = species.e
e.q
e.m

function customP(T, m) 
    P(x,v) = sqrt(sum(v .^2)) * x + m / exp(T)
end

D_D = Distribution(Maxwellian(TD, D.m), D)
D_x = Distribution(customP(Te, e.m), e)
G = Geometry()

typeof([D_D, D_e])
plasma = ElectrostaticPlasm([D_D, D_e], G)

Plasma.solve(plasma, dim=1, GPU=false) 

