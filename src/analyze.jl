# given f(t,x,v) we can compute density, temperature, and all other parameters. It is probably wise to store T, and n, and use that as inputs to functions that compute all parameters.

# TODO maybe plasma should have a phi, and weights field to save the output with the input? 
#Or maybe make a PlasmaSolution struct that stores the plasma and the rest (including lb, ub)

# TODO test this
using Plasma 

TD = 30000 # eV
Te = 10000 # eV

D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D) 
D_e = Distribution(Maxwellian(Te, e.m), e) 

G = Geometry() 

plasma = ElectrostaticPlasma([D_D, D_e], G)

sol = Plasma.solve(plasma, dim=1, GPU=false)

function get_predicts(sol, dx)
    ds = [infimum(d.domain):dx:supremum(d.domain) for d in sol.domains]

    length_x = Int((length(sol.domains)-1)/2 + 1)
    ds_x = [infimum(d.domain):dx:supremum(d.domain) for d in sol.domains[1:length_x]]

    vars = sol.vars
    phi = sol.phi

    # check if it is Es or Bs
    condition = if sol.plasma isa CollisionlessPlasma
                    dict_vars[end-1] || dict_vars[end]
                else
                    dict_vars[end]
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


function plot(sol, dx=0.1)
    predicts = get_predicts(sol, dx)

    for (var, predict) in predicts
        plot_with_makie(var, predict)
    end
 
end

function plot_with_makie(var, predict)
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
end
