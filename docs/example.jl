using Plasma
using BSON: @save

@info "starting two stream instability model"

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

@save "two_stream.bson" sol
