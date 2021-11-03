# Slowly move useful things here, then rename

# Inspiration: https://github.com/SciML/ModelingToolkitStandardLibrary.jl/blob/24e2625ec9b9b045a0b5683e6da4dfeda4ff6fc7/src/Electrical/Analog/ideal_components.jl#L7

# receive Plasma object

# unpack species

# generate variables, bcs for each species

# build neural network for these species

# optimize

# return a sol object with the results + analysis

#=
discretization = NeuralPDE.PhysicsInformedNN(chain, strategy, init_params= initθ)
@named pde_system = PDESystem(eqs, bcs, domains, indvars, depvars)
prob = SciMLBase.discretize(pde_system, discretization)
res = GalacticOptim.solve(prob, opt, cb = cb, maxiters=200)
=#

