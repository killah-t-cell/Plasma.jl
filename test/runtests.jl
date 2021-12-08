using Pkg
using SafeTestsets

@time @safetestset "Plasma tests" begin include("plasmatests.jl")