@info "starting plasma test"

using Plasma
using Test

@info "starting 1D1V test"
TD = 30000
D = species.D
D_D = Distribution(Maxwellian(TD, D.m), D) 
G = Geometry() 
plasma = ElectrostaticPlasma([D_D], G)

sol = Plasma.solve(plasma, dim=1, GPU=false, ub=0.1) 
@test sol isa PlasmaSolution