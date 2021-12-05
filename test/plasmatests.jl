using Plasma
using Test

TD = 30000
D = species.D
D_D = Distribution(Maxwellian(TD, D.m), D) 
G = Geometry() 
plasma = ElectrostaticPlasma([D_D], G)

sol = Plasma.solve(plasma, dim=1, GPU=false, ub=0.1) 
@test sol isa PlasmaSolution