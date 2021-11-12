# Plasma.jl

Plasma.jl is an interface for high-performance simulation of 7D collisionless and electrostatic kinetic plasmas. It solves the full Vlasov and Vlasov-Poisson equations to model plasma evolution with high accuracy.

-Graph-

Knowledge of how plasmas move is essential to solve controlled fusion and understand important astrophysical objects. Knowing how a plasma moves – however – is not trivial. It requires one to know the approximate position and velocity of particles at every point in time. This means the Vlasov equations are generally high dimensional – they must be solved in both velocity and configuration space. Unfortunately, the curse of dimensionality makes solving such 7D equations with standard algorithms infesible with modern technology.

Plasma.jl avoids this by instead using a physics-informed neural network (PINN) architecture to approximate the movement of plasmas. Thereby greatly reducing the cost of computing high-dimensional plasmas.

## Installation

To install Plasma.jl, use the Julia package manager:

```julia
julia> using Pkg
julia> Pkg.add("Plasma")
```

## Features

- Vlasov-Maxwell and Vlasov-Poisson solvers.
- An interface for the definition of plasmas with arbitrary dimensions, species, and initial distributions.
- An interface to define the geometry of a plasma (WIP).
- Plotting functions for easy analysis of results (WIP).

This package is still a work in progress! Some of these features might still have bugs. So feel free to create an issue and we'll try to help you out.

## Example: Solving 3D Electrostatic D-D Plasma

```julia
using Plasma

TD = 30000 # eV
Te = 10000 # eV

D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D)
D_e = Distribution(Maxwellian(Te, e.m), e)
G = Geometry()

plasma = ElectrostaticPlasma([D_D, D_e], G)

sol = Plasma.solve(plasma, dim=2, GPU=false)

Plasma.plot(sol)
```

-Graph-

## Example: Solving 5D Electrostatic α Plasma with a Custom Initial Distribution

```julia
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
```

-Graph-

## Example: Solving 7D Collisionless D-T Plasma

```julia
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
```

-Graph-
