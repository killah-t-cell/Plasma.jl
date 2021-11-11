# given f(t,x,v) we can compute density, temperature, and all other parameters. It is probably wise to store T, and n, and use that as inputs to functions that compute all parameters.

# TODO maybe plasma should have a phi, and weights field to save the output with the input? 
#Or maybe make a PlasmaSolution struct that stores the plasma and the rest (including lb, ub)

# TODO test this
using Plasma 
import ModelingToolkit: Interval, infimum, supremum

TD = 30000 # eV
Te = 10000 # eV

D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D) 
D_e = Distribution(Maxwellian(Te, e.m), e) 

G = Geometry() 

plasma = ElectrostaticPlasma([D_D, D_e], G)

sol = Plasma.solve(plasma, dim=1, GPU=false) 

get_u_predict(sol)

ds = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains]
phi = sol.phi
res = sol.res

using ModelingToolkit
fs = Symbolics.variables(:f, 1:2; T=SymbolicUtils.FnType{Tuple,Real})
Es = Symbolics.variables(:E, 1:1; T=SymbolicUtils.FnType{Tuple,Real})

Dict("f" => phi[1])

sol.domains
ds = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains]
length_x = Int((length(sol.domains)-1)/2 + 1)
ds_x = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains[1:length_x]]

sol.domains

# get right indvars
# when it is 1D1V -> 3f 2E
# when it is 2D2V -> 5f 3E
# when it is 3D3V -> 7f 4E
# when it is 3D3V -> 9f 5E
(length(sol.domains)-1)/2 + 1

# get right depvars
# f, Is, Ivs, E, B 
# f, Is, E
# I need the right length of 

phi



phase_space_vars = [fs;]
configuration_space_vars = [Es;]

Is = Symbolics.variables(:I, eachindex(fs); T=SymbolicUtils.FnType{Tuple,Real})

Is = [1,2]
length(fs)

@parameters t
xs,vs = Symbolics.variables(:x, 1:1), Symbolics.variables(:v, 1:1)

_Es = [E(t,xs...) for E in Es]
_fs = [f(t,xs...,vs...) for f in fs]

bla = [Es;fs]

dict_vars
Es
@show fs[1] ∈ dict_vars[fs] || dict_vars[Es]

pvs = [Es;fs]

vars = [Es, fs, Is]
###
dict_vars = Dict()
for var in vars
    push!(dict_vars, var => [v for v in var])    
end
#####
# vars, phi, domains, dict_vars
function get_predicts(sol)
    ds = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains]

    length_x = Int((length(sol.domains)-1)/2 + 1)
    ds_x = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains[1:length_x]]

    vars = sol.vars
    phi = sol.phi

    condition = if sol.plasma isa CollisionlessPlasma
                    dict_vars[Es] || dict_vars[Bs]
                else
                    dict_vars[Es]
                end
    
    predicts = Dict()
    for i in eachindex(phi)
        if vars[i] ∈ condition
            push!(predicts, vars[i] => [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds_x...)])
        else
            push!(predicts, vars[i] => [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)])
        end
    end
    
    return predicts
end

predicts
###########
for (v, p) in predicts
    @show "title $v"
end
#####

domain, vars
fs

ds = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains]
    
# get x space


length_x = Int((length(sol.domains)-1)/2 + 1)
ds_x = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains[1:length_x]]

f_predicts = []
E_predicts = []
I_predicts = []
Iv_predicts = []
B_predicts = []

for i in eachindex(phi)
    if i <= length(fs)
        push!(f_predicts, [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)])
    elseif i <= length(fs) + length(Is)
        push!(I_predicts, [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)])
    elseif i <= length(fs) + length(Is) + length(Ivs)
        push!(I_predicts, [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)])
    elseif i <= length(fs) + length(Is) + length(Es)
        push!(E_predicts, [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds_x...)])
    end
end

I_predicts
return 
######

for phi in eachindex(phi)
    u_predict = [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)]
    push!(u_predicts, u_predict)
end

fs_phis
Es_phis

plasma.distributions
length(ds)

length_phi_config = length(phase_space_vars):(length(phase_space_vars)+length(configuration_space_vars))
configuration_dict = Dict(var => phi[i] for var in configuration_space_vars for i in length_phi_config)
phi_dicts = [phase_space_dict, configuration_dict]

for i in phase_space_dict
    @show i
end 
# TODO turn phi_dicts into u_predicts

u_predict = [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)]

u_predict = [first(phi[4](collect(d), minimizers_[4])) for d in Iterators.product(ds...)]
u_predicts = []
if phi isa Array
    acum =  [0;accumulate(+, length.(sol.initθ))]
    sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
    minimizers_ = [res.minimizer[s] for s in sep]
    for i in eachindex(phi)
        u_predict = [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)]
        push!(u_predicts, u_predict)
    end
else
    u_predict = [first(phi(collect(d), res.minimizers)) for d in Iterators.product(ds...)]
    push!(u_predicts, u_predict)
end

function get_u_predict(sol::PlasmaSolution)
    ds = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains]
    
    # get x space
    length_x = Int((length(sol.domains)-1)/2 + 1)
    ds_x = [infimum(d.domain):0.1:supremum(d.domain) for d in sol.domains[1:length_x]]
    
    phi = sol.phi
    res = sol.res
    initθ = sol.initθ

    u_predicts = []
    if phi isa Array
        acum =  [0;accumulate(+, length.(initθ))]
        sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
        minimizers_ = [res.minimizer[s] for s in sep]
        for i in eachindex(phi)
            u_predict = [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)]
            push!(u_predicts, u_predict)
        end
    else
        u_predict = [first(phi(collect(d), res.minimizers)) for d in Iterators.product(ds...)]
        push!(u_predicts, u_predict)
    end

    return u_predicts
end

function plot(plasma::CollisionlessPlasma, dx=0.1)
    u_predicts = get_u_predict(plasma)

    if length(plasma.domains) > 3
        # need to give them the length of f, E and B
        # how do I pass the Es and Bs correctly? They are vectors of variable length
        # TODO there can be many fs so this needs to be for f in fs
        f_plot = plot_with_makie(u_predicts[1])
        E_plot = plot_with_makie(u_predicts[4])
        B_plot = plot_with_makie(u_predicts[5])
    else
        plot_with_plots()
    end
end

function plot(plasma::ElectrostaticPlasma, dx=0.1)
    u_predicts = get_u_predict(plasma)

    if length(plasma.domains) > 3
    
    else
        
    end
end

function plot_with_plots()
end

function plot_with_makie()
end

# f plots
# x, v, t -> plots
# x, y, vx, vy, t -> GLMakie
# x, y, z, vx, vy, vz, t -> GLMakie

# E, B plots
# x, t -> Plots
# x, y, t -> Plots
# x, y, z, t -> GLMakie

# function plot(plasma, dx) 
# processes -> u_predict
# if dimensions > 3 -> plot with GLMakie
# =< 3 -> plot with plots
# split each job into separate functions

# Sol.E([0,0.2,0.3]) returns the result of E at these coordinates
# How do I implement that?

# x,y,z,t plotting


# retrieve f
using ModelingToolkit
import ModelingToolkit: Interval, infimum, supremum

@parameters t
xs,vs = Symbolics.variables(:x, 1:2), Symbolics.variables(:v, 1:2)

# domains
xs_int = xs .∈ Interval(0, 1)
vs_int = vs .∈ Interval(0, 1)
t_int = t ∈ Interval(0, 1)

domains = [t_int;xs_int;vs_int]
ds = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]

f(ds...) = ds
fun = collect([f(d1,d2) for d1 in ds[1], d2 in ds[2]]) # returns 11x11 Matrix
fun = collect([phi([d1,d2], res.minimizer) for d1 in ds[1], d2 in ds[2]]) # returns 11x11 Matrix


a = Iterators.product((first.(f.(ds[i])) for i in 1:2)...) |> collect
fun == a
collect(Iterators.product((f.(ds) for _ in 1:2)...))

axis = 0:0.1:1
Iterators.product(axis, axis) |> collect




xs
[[d...] for d in ds]



f_predict = [collect(first(phi[1]([t,x,y,z], res.minimizer[1])) for x in xs, y in xs, z in zs) for t in ts]
E_predict
B_predict


using Makie

ts, xs, ys, zs = [infimum(d.domain):0.1:supremum(d.domain) for d in domains]
u_predict = [collect(phi([t,x,y,z], res.minimizer)[1] for x in xs, y in xs, z in zs) for t in ts]

positions = Node(u_predict[1])
scene = volume(xs, ys, zs, positions, colormap = :plasma, colorrange = (minimum(vol), maximum(vol)),figure = (; resolution = (800,800)),  
                axis = (; type=Axis3, perspectiveness = 0.5,  azimuth = 7.19, elevation = 0.57,  
                aspect = (1,1,1)))

fps = 60
record(scene, "output.mp4", eachindex(ts)) do t
    positions[] = u_predict[t]
    sleep(1/fps)
end

# x,y,z,vx,vy,vz, t

using GLMakie
GLMakie.activate!()

ts = 1:10
i = Observable(1)

u_predict

f = Figure()

data_1 = @lift(u_predict[$i][:, :, :, 1, 1, 1])
data_2 = @lift(u_predict[$i][1, 1, 1, :, :, :])

GLMakie.volume(f[1, 1], data_1, axis = (;type = Axis3))
GLMakie.volume(f[1, 2], data_2, axis = (;type = Axis3))

ls = labelslider!(f, "t", ts)
f[2, 1:2] = ls.layout
connect!(i, ls.slider.value)

f

# x,y,vx,vy,t

using GLMakie
GLMakie.activate!()

ts = 1:10
i = Observable(1)

u_predict

f = Figure()

data_1 = @lift(u_predict[$i][:, :, 1, 1])
data_2 = @lift(u_predict[$i][1, 1, :, :])

GLMakie.volume(f[1, 1], data_1, axis = (;type = Axis3)) # this ought to be different
GLMakie.volume(f[1, 2], data_2, axis = (;type = Axis3))

ls = labelslider!(f, "t", ts)
f[2, 1:2] = ls.layout
connect!(i, ls.slider.value)

f


#
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C=1
eq  = Dtt(u(t,x)) ~ C^2*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(t,0) ~ 0.,# for all t > 0
       u(t,1) ~ 0.,# for all t > 0
       u(0,x) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(0,x)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]
# Discretization
dx = 0.1

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
discretization = PhysicsInformedNN(chain, GridTraining(dx); init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# optimizer
opt = Optim.BFGS()
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=1200)
phi = discretization.phi

using Plots


f(ds...) = ds
ds = [0.0:0.1:1.0 for d in 1:2]

ds_ = [[d for d in ds[i]] for i in 1:length(ds)]


fun = collect([f(d...) for d in ds[1], d2 in ds[2]])


plot(ds..., u_predict, linetype=:contourf,title = "predict")



u_predict = collect([first(phi([t,x],res.minimizer)) for t in ts, x in xs])


[d...] for d in ds[1], d in ds[2]


ds
[[d...] for d in ds]



[[d for d in ds[i]] for i in 1:length(ds)]




using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum
using Plots

@parameters t, x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C=1
eq  = Dtt(u(t,x)) ~ C^2*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(t,0) ~ 0.,# for all t > 0
       u(t,1) ~ 0.,# for all t > 0
       u(0,x) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(0,x)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(0.0,1.0)]
# Discretization
dx = 0.1

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
discretization = PhysicsInformedNN(chain, GridTraining(dx); init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[t,x],[u(t,x)])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

# optimizer
opt = Optim.BFGS()
res = GalacticOptim.solve(prob,opt; cb = cb, maxiters=40)
phi = discretization.phi
ds = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
ts,xs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]


# THIS IS THE WAY
u_predict = [first(phi(collect(d), res.minimizer)) for d in Iterators.product(ds...)]
u_predict2 = reshape([first(phi([t,x],res.minimizer)) for t in ts for x in xs],(length(ts),length(xs)))

f.(Iterators.product(ds...))