##### WITH DOMAIN DECOMPOSITION
using ModelingToolkit
using Flux
using NeuralPDE
using GalacticOptim
using DiffEqFlux
using DomainSets
import ModelingToolkit: Interval, infimum, supremum
using Plots
using CUDA

@parameters t x v
@variables f(..) E(..) 
Dx = Differential(x)
Dt = Differential(t)
Dv = Differential(v)

# Constants
μ_0 = 1.25663706212e-6 # N A⁻²
ε_0 = 8.8541878128e-12 # F ms⁻¹
e   = 1.602176634e-19 # Coulombs
m_e = 9.10938188e-31 # Kg
n_0 = 1
v_th = sqrt(2)

# Space
t_min = 0.0
t_max = 1.0
x_min = 0.0
x_max = 1.0
v_min = 0.0
v_max = 1.0
t_domain = Interval(t_min, t_max)
x_domain = Interval(x_min, x_max)
v_domain = Interval(v_min, v_max)
length_decomposition = 0.1

count_decomp = Int16(((t_max-t_min)/length_decomposition))
domains = [t ∈ t_domain,
           x ∈ x_domain, 
           v ∈ v_domain]

# Equations
eqs = [Dt(f(t,x,v)) ~ - v * Dx(f(t,x,v)) - e/m_e * E(t,x) * Dv(f(t,x,v))
       Dx(E(t,x)) ~ e*n_0/ε_0 * (Dt(f(t,x,v)) - 1)]


function set_boundaries(t_0)
    bcs = [f(t_0,x,v) ~ 1/(v_th * sqrt(2π)) * exp(-v^2/(2*v_th^2)), 
           E(t_0,x) ~ e*n_0/ε_0 * (Dt(f(0,x,v)) - 1)]
end

bcs = set_boundaries(0)

# Neural Network
chains = [[FastChain(FastDense(3, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1)), 
          FastChain(FastDense(2, 16, Flux.σ), FastDense(16,16,Flux.σ), FastDense(16, 1))] for _ in 1:count_decomp]           
initθ = []
initθ_ = [DiffEqFlux.initial_params.(ch) for ch in chains]
for i in 1:length(initθ_)
    push!(initθ, map(c -> Float64.(c), initθ_[i]))
end
initθ

ts_ = infimum(t_domain):1/count_decomp:supremum(t_domain)
ts_domain = [(ts_[i], ts_[i+1]) for i in 1:length(ts_)-1]
domains_map = map(ts_domain) do (ts_dom)
    t_domain_ = Interval(ts_dom...)
    domains_ = [t ∈ t_domain_,
                x ∈ x_domain,
                v ∈ v_domain]
end

#=
function create_bcs(t_domain_)
    t_0 = t_domain_.left
    set_boundaries(t_0)
end
=#
function create_bcs(bcs,t_domain_,phi_bound)
    t_0, t_e =  t_domain_.left, t_domain_.right
    if t_0 == 0.0
        bcs = [f(t_0,x,v) ~ set_initial_geometry(x) * 1/(v_th * sqrt(2π)) * exp(-v^2/(2*v_th^2)), 
               E(t_0,x) ~ set_initial_geometry(x) * e*n_0/ε_0 * (Iv(f(0,x,v)) - 1)]
        return bcs
    end
    bcs = [f(t_0,x,v) ~ phi_bound(t_0,x,v),
           E(t_0,x) ~ set_initial_geometry(x) * e*n_0/ε_0 * (Iv(f(0,x,v)) - 1)]
    bcs
end

reses = []
phis = []
pde_system_map = []

# cb
cb = function (p,l)
    println("Current loss is: $l")
    return false
end

for i in 1:count_decomp
    println("decomposition $i")
    domains_ = domains_map[i]

    phi_in(cord) = phis[i-1](cord,reses[i-1].minimizer)
    phi_bound(t,x,v) = phi_in(vcat(t,x,v))
    @register phi_bound(t,x,v)
    Base.Broadcast.broadcasted(::typeof(phi_bound), t,x,v) = phi_bound(t,x,v)
    bcs_ = create_bcs(domains_[1].domain)
    @named pde_system_ = PDESystem(eqs, bcs_, domains_, [t,x,v], [f(t,x,v), E(t,x)])
    push!(pde_system_map,pde_system_)
    strategy = NeuralPDE.QuadratureTraining()

    discretization = NeuralPDE.PhysicsInformedNN(chains[i], strategy; init_params=initθ[i]) 

    prob = NeuralPDE.discretize(pde_system_,discretization)
    symprob = NeuralPDE.symbolic_discretize(pde_system_,discretization)
    res_ = GalacticOptim.solve(prob, BFGS(); cb=cb, maxiters=100)
    phi = discretization.phi
    push!(reses, res_)
    push!(phis, phi)
end

function compose_result(dx)
    u_predict_array_f = Float64[]
    u_predict_array_E = Float64[]
    diff_u_array = Float64[]
    xs = infimum(domains[2].domain):dx:supremum(domains[2].domain)
    vs = infimum(domains[3].domain):dx:supremum(domains[3].domain)
    ts_ = infimum(t_domain):dx:supremum(t_domain)
    ts = collect(ts_)
    function index_of_interval(t_)
        for (i,t_domain) in enumerate(ts_domain)
            if t_<= t_domain[2] && t_>= t_domain[1]
                return i
            end
        end
    end
    for t_ in ts
        i = index_of_interval(t_)
        acum =  [0;accumulate(+, length.(initθ[i]))]
        sep = [acum[j]+1 : acum[j+1] for j in 1:length(acum)-1]
        minimizers_ = [reses[i].minimizer[s] for s in sep]
        u_predict_sub_f = [first(phis[i][1]([t_,x, v],minimizers_[1])) for x in xs for v in vs]
        u_predict_sub_E = [first(phis[i][2]([t_,x],minimizers_[2])) for x in xs]
        append!(u_predict_array_f,u_predict_sub_f)
        append!(u_predict_array_E,u_predict_sub_E)
    end
    ts,xs,vs = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
    u_predict_f = reshape(u_predict_array_f,(length(ts),length(xs),length(vs)))
    u_predict_E = reshape(u_predict_array_E,(length(ts),length(xs)))
    u_predict_f, u_predict_E
end
dx= 0.01
u_predict_f, u_predict_E = compose_result(dx)

@show u_predict_E
# Neural adapter
inner_ = 18
af = Flux.tanh
chain2 = [FastChain(FastDense(3,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,1)), 
          FastChain(FastDense(2,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,inner_,af),
                   FastDense(inner_,1))]


initθ2 = map(c -> Float64.(c), DiffEqFlux.initial_params.(chain2))

@named pde_system = PDESystem(eqs, bcs, domains, [t,x,v], [f(t,x,v), E(t,x)])

# the error is most likely here
minimizers_ = [reses[i].minimizer[s] for s in sep]
u_predict_sub_f = [first(phis[i][1]([t_,x, v],minimizers_[1])) for x in xs for v in vs]
u_predict_sub_E = [first(phis[i][2]([t_,x],minimizers_[2])) for x in xs]

losses = map(1:count_decomp) do i
    minimizers_ = [reses[i].minimizer[s] for s in sep]
    loss(cord,θ) = chain2[1](cord,θ) .- phis[i](cord,minimizers_[1])
end

pde_system_map[3].bcs
pde_system_map[3].bcs


prob_ = NeuralPDE.neural_adapter(losses,initθ2, pde_system_map,NeuralPDE.GridTraining([0.1/count_decomp,0.1]))
res_ = GalacticOptim.solve(prob_, ADAM(0.01);cb=cb, maxiters=2000)
prob_ = NeuralPDE.neural_adapter(losses,res_.minimizer, pde_system_map, NeuralPDE.GridTraining([0.05/count_decomp,0.05]))
res_ = GalacticOptim.solve(prob_, BFGS();cb=cb,  maxiters=1000)

parameterless_type_θ = DiffEqBase.parameterless_type(initθ2)
phi_ = NeuralPDE.get_phi(chain2,parameterless_type_θ)
