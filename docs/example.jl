using Plasma
using Plots

TD = 0.3
Te = 0.1
D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D)
D_e = Distribution(Maxwellian(Te, e.m), e) 
G = Geometry() 

plasma = ElectrostaticPlasma([D_D, D_e], G)

sol = Plasma.solve(plasma, dim=1, GPU=false) 

Plasma.plot(sol)

## Two-stream instability

e = species.e

function TwoStream(vth2, vs1, vs2) 

    function P(x,v)
        if !(v isa Array)
            v = [v]    
        end

        if !(x isa Array)
            x = [x]    
        end

        v = sqrt(sum(v .^2))
        x = sqrt(sum(x .^2))

        0.5/sqrt(vth2 * π) * exp(-(v-vs1)*(v-vs1)/vth2) + 0.5/sqrt(vth2 * π) * exp(-(v-vs2)*(v-vs2)/vth2) * (1+0.02*cos(3*π*x))
    end
end

D_e = Distribution(TwoStream(0.02, 1.6, -1.4), e) 
G = Geometry() 

plasma = ElectrostaticPlasma([D_e], G)

sol = Plasma.solve(plasma, dim=1, GPU=false, time_ub = 100.0, ub=2.0) 

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
