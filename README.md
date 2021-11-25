# Plasma.jl

Plasma.jl is an interface for high-performance simulation of 7D collisionless and electrostatic kinetic plasmas. It solves the full [Vlasov-Maxwell and Vlasov-Poisson equations](https://en.wikipedia.org/wiki/Vlasov_equation) to model plasma evolution with high accuracy.

## Problem Domain

Knowledge of how plasmas move is essential to solving controlled nuclear fusion and understanding important astrophysical systems. To know how a plasma moves one must know the approximate position and velocity of particles at every point in time. This is not trivial. In fact, the Vlasov equations non-linearity and high dimensionality makes it infesible to solve such equations with standard algorithms.

Plasma.jl handles this by instead using a [Physics-informed Neural Network (PINN)](https://arxiv.org/abs/2107.09443) architecture to approximate plasma evolution. Thereby greatly reducing the cost of computing high-dimensional plasmas.

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
- Plotting functions for easy analysis of results.
- Validation methods to evaluate error in models (WIP).
- Distributed GPU support (WIP).

**Parts of this package are still work in progress.** Some of these features might still have bugs. So feel free to create an issue and we'll try to help you out.

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
