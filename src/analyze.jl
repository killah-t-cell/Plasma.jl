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

    # checks if it is Es or Bs
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
