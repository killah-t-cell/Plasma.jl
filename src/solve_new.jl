# Slowly move useful things here, then rename

# Inspiration: https://github.com/SciML/ModelingToolkitStandardLibrary.jl/blob/24e2625ec9b9b045a0b5683e6da4dfeda4ff6fc7/src/Electrical/Analog/ideal_components.jl#L7

# receive Plasma object


function solve(plasma::CollisionlessPlasma; lb, ub, time_lb=lb, time_ub=ub, GPU=true, inner_layers=16)
    if lb > ub
        error("lower bound must be larger than upper bound")
    end

    # constants
    dim = 3
    species = plasma.species
    geometry = plasma.geometry
    consts = Constants()
    μ_0, ϵ_0 = consts.μ_0, consts.ϵ_0

    # variables
    fs = Symbolics.variables(:f, eachindex(species); T=SymbolicUtils.FnType{Tuple,Real})
    Es = Symbolics.variables(:E, 1:dim; T=SymbolicUtils.FnType{Tuple,Real})
    Bs = Symbolics.variables(:B, 1:dim; T=SymbolicUtils.FnType{Tuple,Real})

    # integrals
    Ivs = Symbolics.variables(:Iv, eachindex(fs), 1:dim ; T=SymbolicUtils.FnType{Tuple,Real})
    Is = Symbolics.variables(:I, eachindex(fs); T=SymbolicUtils.FnType{Tuple,Real})
    _I = Integral(tuple(vs...) in DomainSets.ProductDomain(ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf)))

    # parameters
    @parameters t
    xs,vs = Symbolics.variables(:x, 1:dim), Symbolics.variables(:v, 1:dim)

    # differentials
    Dxs = Differential.(xs)
    Dvs = Differential.(vs)
    Dt = Differential(t)

    # get qs, ms, Ps from species
    qs, ms, Ps = [], [], []
    for s in species
        push!(qs,s.q)
        push!(ms,s.m)
        push!(Ps,s.P)
    end

    # domains
    xs_int = xs .∈ Interval(lb, ub)
    vs_int = vs .∈ Interval(lb, ub)
    t_int = t ∈ Interval(time_lb, time_ub)

    domains = [t_int;xs_int;vs_int]

    # helpers
    _Es = [E(t,xs...) for E in Es]
    _Bs = [B(t,xs...) for B in Bs]
    _fs = [f(t,xs...,vs...) for f in fs]
    _Is = [I(t,xs...,vs...) for I in Is]
    _Ivs = [Iv(t,xs...,vs...) for Iv in Ivs]

    # divergences
    div_vs = [divergence(Dxs, _f, vs) for _f in _fs]
    div_B = divergence(Dxs, _Bs)
    div_E = divergence(Dxs, _Es)
    Fs = [qs[i]/ms[i] * (_Es + cross(vs,_Bs)) for i in eachindex(qs)]
    divv_Fs = [divergence(Dvs, _fs[i], Fs[i]) for i in eachindex(_fs)]

    # charge and current densities
    ρ = sum([qs[i] * _Is[i] for i in eachindex(qs)])
    J = [sum(qs[i] * _Ivs[i, j] for i in eachindex(qs)) for j in 1:length(eachcol(_Ivs))] 

    # system of equations
    vlasov_eqs = Dt.(_fs) .~ .- div_vs .- divv_Fs
    curl_E_eqs = curl(_Es, Dxs) .~ Dt.(_Bs)
    curl_B_eqs = ϵ_0*μ_0 * Dt.(_Es) .- curl(_Bs, Dxs) .~ - μ_0.*J
    div_E_eq = div_E ~ ρ/ϵ_0
    div_B_eq = div_B ~ 0
    eqs = [vlasov_eqs; curl_E_eqs; curl_B_eqs; div_E_eq; div_B_eq]

    # boundary and initial conditions
    vlasov_ics = [fs[i](0,xs...,vs...) ~ Ps[i] * geometry(xs...) for i in eachindex(fs)]
    div_B_ic = div_B ~ 0
    div_E_ic = div_E ~ sum([qs[i] * Is[i](0,xs...,vs...) for i in eachindex(qs)])/ϵ_0 * geometry(xs...)
    
    bcs_ = [vlasov_ics; div_B_ic; div_E_ic]

    # neural integral
    f_ints = _I.(_fs) .~ _Is 
    vf_ints = [_I(_fs[i]) * vs[j] ~ _Ivs[i, j] for i in 1:length(eachrow(_Ivs)), j in 1:length(eachcol(_Ivs))] 

    ints_ = [f_ints; vcat(vf_ints...)]

    # set up and return PDE System
    bcs = [bcs_;ints_]
    vars = [_fs...; _Is...; _Ivs...; _Es...; _Bs...]
    @named pde_system = PDESystem(eqs, bcs, domains, [t,xs...,vs...], vars)

    # solve
    ps_chains = [FastChain(FastDense(length(domains), il, Flux.σ), FastDense(il,il,Flux.σ), FastDense(il, 1)) for _ in 1:length([_fs; _Is; vcat(_Ivs...)])]
    xs_chains = [FastChain(FastDense(length([t, xs...]), il, Flux.σ), FastDense(il,il,Flux.σ), FastDense(il, 1)) for _ in 1:length([_Es; _Bs])]
    chain = [ps_chains;xs_chains]
    initθ = GPU ? map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain)) : map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) 
    discretization = NeuralPDE.PhysicsInformedNN(chain, QuadratureTraining(), init_params=initθ)
    prob = SciMLBase.discretize(pde_system, discretization)
    
    # Solve
    opt = Optim.BFGS()
    res = GalacticOptim.solve(prob, opt, cb = print_loss(prob), maxiters=200)
    prob = remake(prob, u0=res.minimizer)
    res = GalacticOptim.solve(prob, ADAM(0.01), cb = print_loss(prob), maxiters=10000)
    prob = remake(prob, u0=res.minimizer)
    res = GalacticOptim.solve(prob, opt, cb = print_loss(prob), maxiters=200)
    phi = discretization.phi
    return phi, res, initθ
end


function solve(plasma::ElectrostaticPlasma; lb, ub, dim=3, time_lb=lb, time_ub=ub, GPU=true)
    
end

function curl(vec, Ds)
    [Ds[2](vec[3]) - Ds[3](vec[2]), Ds[3](vec[1]) - Ds[1](vec[3]), Ds[1](vec[2]) - Ds[2](vec[1])]
end

"""
Get the divergence of a function f w.r.t Ds with the option of multiplying each part of the sum by a v
"""
function divergence(Ds, f, v=ones(length(Ds)))
    if f isa AbstractArray
        sum([v[i] * Ds[i](f[i]) for i in eachindex(Ds)])
    else
        sum([v[i] * Ds[i](f) for i in eachindex(Ds)])
    end
end

function print_loss(prob)
    pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
    inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents
    bcs_inner_loss_functions = inner_loss_functions[1:4]
    aprox_integral_loss_functions = inner_loss_functions[5:end]

    cb = function (p,l)
    println("Current loss is: $l")
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
    end
end


