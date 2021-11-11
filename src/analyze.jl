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

predicts = get_predicts(sol, 0.1)

plot_plasma(sol)

for (i,predict) in predicts
    ts = 1:10
    GLMakie.activate!()

    i = Observable(1)
    f = Figure()

    data_1 = @lift(predict[$i, :, :])

    ls = labelslider!(f, "t", ts)
    # 1 means 1 graph, 1:2 means 2 graphs
    f[2, 1:2] = ls.layout
    connect!(i, ls.slider.value)

    Makie.save("test.png", f)
end

ts = length(0.0:0.1:1.0)
typeof(0:1)

vcat(0:1)

function get_predicts(sol, dx)
    ds = [infimum(d.domain):dx:supremum(d.domain) for d in sol.domains]

    length_x = Int((length(sol.domains)-1)/2 + 1)
    ds_x = [infimum(d.domain):dx:supremum(d.domain) for d in sol.domains[1:length_x]]

    acum =  [0;accumulate(+, length.(sol.initθ))]
    sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
    minimizers_ = [sol.res.minimizer[s] for s in sep]

    vars = sol.vars
    phi = sol.phi
    dict_vars = sol.dict_vars

    # check if it is Es or Bs
    condition = if sol.plasma isa CollisionlessPlasma
                    dict_vars[vars[end-1]] || dict_vars[vars[end]]
                else
                    dict_vars[vars[end]]
                end
    
    predicts = Dict()

    i_vars = vcat(vars...)
    for i in eachindex(phi)
        if i_vars[i] ∈ condition
            push!(predicts, i_vars[i] => [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds_x...)])
        else
            push!(predicts, i_vars[i] => [first(phi[i](collect(d), minimizers_[i])) for d in Iterators.product(ds...)])
        end
    end
    
    return predicts
end

function plot_plasma(sol::PlasmaSolution, dx=0.1)
    predicts = get_predicts(sol, dx)

    for (var, predict) in predicts
        plot_with_makie(var, predict, ds[1])
    end
 
end

function plot_with_makie(var, predict, ts)
    ts = 1:length(ts)
    GLMakie.activate!()

    # 1 means 1 graph, 1:2 means 2 graphs
    multigraph = length(size(predict)) > 4 ? (1:2) : 1

    i = Observable(1)
    f = Figure()

    if length(size(predict)) == 7
        data_1 = @lift(predict[$i, :, :, :, 1, 1, 1])
        data_2 = @lift(predict[$i, 1, 1, 1, :, :, :])

        volume(f[1, 1], data_1, axis = (;type = Axis3))
        volume(f[1, 2], data_2, axis = (;type = Axis3))

    elseif length(size(predict)) == 5
        data_1 = @lift(predict[$i, :, :, 1, 1])
        data_2 = @lift(predict[$i, 1, 1, :, :])

        heatmap(f[1, 1], data_1)
        heatmap(f[1, 2], data_2)

    elseif length(size(predict)) == 4
        data_1 = @lift(predict[$i, :, :, :])
        volume(f[1, 1], data_1, axis = (;type = Axis3))

    elseif length(size(predict)) == 3
        data_1 = @lift(predict[$i, :, :])
        heatmap(f[1, 1], data_1)

    elseif length(size(predict)) == 2
        heatmap(f[1, 1], predict[:, :])
    else
        error("dimension not yet implemented.")
    end

    ls = labelslider!(f, "t", ts)
    
    f[2, multigraph] = ls.layout
    connect!(i, ls.slider.value)

    Makie.save("$var.png", f)
end


heatmap(rand(120, 120), algorithm = :mip)

using Makie
GLMakie.activate!()

ts = 1:0.1:10
data = rand(10,10,10,10,10,10,10)

i = Observable(1)

f = Figure()

data_1 = @lift(data[$i, :, :, :, 1, 1, 1])

volume(f[1, 1], data_1, axis = (;type = Axis3))

ls = labelslider!(f, "t", ts)
f[2, 1:2] = ls.layout
connect!(i, ls.slider.value)

f