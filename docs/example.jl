using Plasma
using Plots
using JSON
using NeuralPDE
import ModelingToolkit: Interval, infimum, supremum

TD = 0.3
Te = 0.2
D = species.D
e = species.e

D_D = Distribution(Maxwellian(TD, D.m), D)
D_e = Distribution(Maxwellian(Te, e.m), e) 
G = Geometry() 

plasma = ElectrostaticPlasma([D_e], G)

sol = Plasma.solve(plasma, dim=1, GPU=false, strategy=QuadratureTraining(), maxiters=[80, 1], time_ub = 5) 
sol.res
data = Dict("res"=>sol.res)
json_string = JSON.json(data)

open("model_march_quadrature_5t_maxwellian.json","w") do f
    JSON.print(f, json_string, 4)
end

# plot
vcat(sol.vars...)
f_predict = Plasma.get_predicts(sol, 0.01)[vcat(sol.vars...)[1]]
ts, xs, vs = [infimum(sol.domains[d].domain):0.01:supremum(sol.domains[d].domain) for d in 1:length(sol.domains)]

f_predict[4, :, :]
function plot_f()
    anim = @animate for t in eachindex(ts)
        @info "Animating frame $t..."
        p1 = heatmap(xs, vs, f_predict[t, :, :], label="$t", title="f_$t")
        plot(p1)
    end
    gif(anim,"f_quadrature_maxwellian.gif", fps=30)
end
plot_f()

# save
data = Dict("sol"=>sol)
json_string = JSON.json(data)

open("f_quadrature_maxwellian_model_sol.json","w") do f
    JSON.print(f, json_string, 4)
end

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
