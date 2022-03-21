# Plasma.jl

Plasma.jl is an interface for high-performance simulations of 7D collisionless and electrostatic kinetic plasmas. It models plasma evolution by solving the full [Vlasov-Maxwell and Vlasov-Poisson equations](https://en.wikipedia.org/wiki/Vlasov_equation).

## Problem Domain

Knowing how plasmas move is essential to solving controlled nuclear fusion and understanding important astrophysical systems. Plasma movement can be approximated by knowing the distribution function of particles averaged over the Debye length. This is not trivial. In fact, the Vlasov equation's non-linearity and high dimensionality makes it unfeasible to solve such equations with standard mesh or PIC algorithms.

Plasma.jl handles this by instead using a [Physics-informed Neural Network (PINN)](https://arxiv.org/abs/2107.09443) architecture to approximate plasma evolution. Thereby reducing the cost of computing high-dimensional plasmas.

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

TD = 0.3 # eV
Te = 0.2 # eV
D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D)
D_e = Distribution(Maxwellian(Te, e.m), e) 
G = Geometry() 

plasma = ElectrostaticPlasma([D_D, D_e], G)

sol = Plasma.solve(plasma, dim=1, GPU=false, strategy=QuadratureTraining()) 
```

## Example: Solving 5D Electrostatic α Plasma with a Custom Initial Distribution

```julia
## 2D with custom P and species
using Plasma

Tα = 70000 # eV

α = Species(1.602176634e-19, 6.6446562e-27)

function HotCarrier(T) 
    Kb = 8.617333262145e-5
    function P(x,v)
        v_ = sqrt(sum(v .^2))
        exp(-v_/(Kb*T))
    end
end

Dα = Distribution(HotCarrier(Tα), α)
G = Geometry()

plasma = ElectrostaticPlasma([Dα], G)

sol = Plasma.solve(plasma, dim=2) # with GPU

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

## Plotting
You can build your own plotting functions, or check out `src/analyze.jl` for a fairly crude way to plot models with Makie. A Makie recipe is a WIP right now.

## Benchmarks

See [PlasmaBenchmarks.jl](https://github.com/killah-t-cell/PlasmaBenchmarks.jl) for solved examples and benchmarks against PIC methods w.r.t. performance and accuracy.

