"""
Solve dispatch for collisionless plasmas
"""
function solve(plasma::CollisionlessPlasma; 
               lb=0.0, ub=1.0, time_lb=lb, time_ub=ub, 
               GPU=true, inner_layers=16, strategy=QuadratureTraining())
    if lb > ub
        error("lower bound must be larger than upper bound")
    end

    # constants
    dim = 3
    geometry = plasma.geometry.f # this might change with a geometry refactor
    dis = plasma.distributions
    species = [d.species for d in dis]
    consts = Constants()
    μ_0, ϵ_0 = consts.μ_0, consts.ϵ_0
    
    # get qs, ms, Ps from species
    qs, ms = [], []
    for s in species
        push!(qs,s.q)
        push!(ms,s.m)
    end
    Ps = [d.P for d in dis]

    # variables
    fs = Symbolics.variables(:f, eachindex(species); T=SymbolicUtils.FnType{Tuple,Real})
    Es = Symbolics.variables(:E, 1:dim; T=SymbolicUtils.FnType{Tuple,Real})
    Bs = Symbolics.variables(:B, 1:dim; T=SymbolicUtils.FnType{Tuple,Real})

    # parameters
    @parameters t
    xs,vs = Symbolics.variables(:x, 1:dim), Symbolics.variables(:v, 1:dim)

    # integrals
    Ivs = Symbolics.variables(:Iv, eachindex(fs), 1:dim ; T=SymbolicUtils.FnType{Tuple,Real})
    Is = Symbolics.variables(:I, eachindex(fs); T=SymbolicUtils.FnType{Tuple,Real})
    _I = Integral(tuple(vs...) in DomainSets.ProductDomain(ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf), ClosedInterval(-Inf ,Inf)))    

    # differentials
    Dxs = Differential.(xs)
    Dvs = Differential.(vs)
    Dt = Differential(t)

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
    vlasov_ics = [fs[i](0,xs...,vs...) ~ Ps[i](xs,vs) * geometry(xs) for i in eachindex(fs)]
    div_B_ic = div_B ~ 0
    div_E_ic = div_E ~ sum([qs[i] * Is[i](0,xs...,vs...) for i in eachindex(qs)])/ϵ_0 * geometry(xs)
    
    bcs_ = [vlasov_ics; div_B_ic; div_E_ic]

    # neural integral
    f_ints = _I.(_fs) .~ _Is 
    vf_ints = [_I(_fs[i]) * vs[j] ~ _Ivs[i, j] for i in 1:length(eachrow(_Ivs)), j in 1:length(eachcol(_Ivs))] 

    ints_ = [f_ints; vcat(vf_ints...)]

    # set up variables # TODO turn this into a separate function
    bcs = [bcs_;ints_]
    vars_arg = [_fs...; _Is...; _Ivs...; _Es...; _Bs...]
    vars = [fs, Is, Ivs, Bs, Es]
    
    dict_vars = Dict()
    for var in vars
        push!(dict_vars, var => [v for v in var])    
    end

    # set up PDE System
    @named pde_system = PDESystem(eqs, bcs, domains, [t,xs...,vs...], vars_arg)

    # set up problem
    il = inner_layers
    ps_chains = [FastChain(FastDense(length(domains), il, Flux.σ), FastDense(il,il,Flux.σ), FastDense(il, 1)) for _ in 1:length([_fs; _Is; vcat(_Ivs...)])]
    xs_chains = [FastChain(FastDense(length([t, xs...]), il, Flux.σ), FastDense(il,il,Flux.σ), FastDense(il, 1)) for _ in 1:length([_Es; _Bs])]
    chain = [ps_chains;xs_chains]
    initθ = GPU ? map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain)) : map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) 
    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy, init_params=initθ)
    prob = SciMLBase.discretize(pde_system, discretization)
    
    # solve
    opt = Optim.BFGS()
    res = GalacticOptim.solve(prob, opt, cb = print_loss(prob), maxiters=200)
    prob = remake(prob, u0=res.minimizer)
    res = GalacticOptim.solve(prob, ADAM(0.01), cb = print_loss(prob), maxiters=10000)
    prob = remake(prob, u0=res.minimizer)
    res = GalacticOptim.solve(prob, opt, cb = print_loss(prob), maxiters=200)
    phi = discretization.phi


    return PlasmaSolution(plasma, vars, dict_vars, phi, res, initθ, domains)
end

"""
Solve dispatch for electrostatic plasmas
"""
function solve(plasma::ElectrostaticPlasma; 
    lb=0.0, ub=1.0, time_lb=lb, time_ub=ub, 
    dim=3, GPU=true, inner_layers=16, strategy=QuadratureTraining())
    if lb > ub
        error("lower bound must be larger than upper bound")
    end

    # constants
    geometry = plasma.geometry.f # this might change with a geometry refactor
    dis = plasma.distributions
    species = [d.species for d in dis]
    consts = Constants()
    μ_0, ϵ_0 = consts.μ_0, consts.ϵ_0

    # get qs, ms, Ps from species
    qs, ms = [], []
    for s in species
        push!(qs,s.q)
        push!(ms,s.m)
    end
    Ps = [d.P for d in dis]

    # variables
    fs = Symbolics.variables(:f, eachindex(species); T=SymbolicUtils.FnType{Tuple,Real})
    Es = Symbolics.variables(:E, 1:dim; T=SymbolicUtils.FnType{Tuple,Real})

    # parameters
    @parameters t
    xs,vs = Symbolics.variables(:x, 1:dim), Symbolics.variables(:v, 1:dim)

    # integrals
    Is = Symbolics.variables(:I, eachindex(fs); T=SymbolicUtils.FnType{Tuple,Real})
    if length(vs) > 1
        intervals = [ClosedInterval(-Inf ,Inf) for _ in 1:length(vs)]
        _I = Integral(tuple(vs...) in DomainSets.ProductDomain(intervals...))
    else
        _I = Integral(first(vs) in DomainSets.ClosedInterval(-Inf ,Inf))
    end

    # differentials
    Dxs = Differential.(xs)
    Dvs = Differential.(vs)
    Dt = Differential(t)


    # domains
    xs_int = xs .∈ Interval(lb, ub)
    vs_int = vs .∈ Interval(lb, ub)
    t_int = t ∈ Interval(time_lb, time_ub)

    domains = [t_int;xs_int;vs_int]

    # helpers
    _Es = [E(t,xs...) for E in Es]
    _fs = [f(t,xs...,vs...) for f in fs]
    _Is = [I(t,xs...,vs...) for I in Is]

    # divergences
    div_vs = [divergence(Dxs, _f, vs) for _f in _fs]
    div_E = divergence(Dxs, _Es)
    Fs = [qs[i]/ms[i] * _Es for i in eachindex(qs)]
    divv_Fs = [divergence(Dvs, _fs[i], Fs[i]) for i in eachindex(_fs)]

    # charge density
    ρ = sum([qs[i] * _Is[i] for i in eachindex(qs)])

    # equations
    vlasov_eqs = Dt.(_fs) .~ .- div_vs .- divv_Fs
    div_E_eq = div_E ~ ρ/ϵ_0
    eqs = [vlasov_eqs; div_E_eq]

    # boundary and initial conditions
    vlasov_ics = [fs[i](0,xs...,vs...) ~ Ps[i](xs,vs) * geometry(xs) for i in eachindex(fs)]
    div_E_ic = div_E ~ sum([qs[i] * Is[i](0,xs...,vs...) for i in eachindex(qs)])/ϵ_0 * geometry(xs) 
    # TODO does E need boundary conditions?

    bcs_ = [vlasov_ics; div_E_ic]

    # neural integral
    ints_ = _I.(_fs) .~ _Is 

    # set up variables # TODO turn this into a separate function
    bcs = [bcs_;ints_]
    vars_arg = [_fs; _Is; _Es]
    vars = [fs, Is, Es]
    
    dict_vars = Dict()
    for var in vars
        push!(dict_vars, var => [v for v in var])    
    end

    # set up and return PDE System
    @named pde_system = PDESystem(eqs, bcs, domains, [t,xs...,vs...], vars_arg)

    # set up problem
    il = inner_layers
    ps_chains = [FastChain(FastDense(length(domains), il, Flux.σ), FastDense(il,il,Flux.σ), FastDense(il, 1)) for _ in 1:length([_fs; _Is])]
    xs_chains = [FastChain(FastDense(length([t, xs...]), il, Flux.σ), FastDense(il,il,Flux.σ), FastDense(il, 1)) for _ in 1:length(_Es)]
    chain = [ps_chains;xs_chains]
    initθ = GPU ? map(c -> CuArray(Float64.(c)), DiffEqFlux.initial_params.(chain)) : map(c -> Float64.(c), DiffEqFlux.initial_params.(chain)) 
    discretization = NeuralPDE.PhysicsInformedNN(chain, strategy, init_params=initθ)
    prob = SciMLBase.discretize(pde_system, discretization)
    
    # solve
    opt = Optim.BFGS()
    res = GalacticOptim.solve(prob, opt, cb = print_loss(prob), maxiters=200)
    prob = remake(prob, u0=res.minimizer)
    res = GalacticOptim.solve(prob, ADAM(0.01), cb = print_loss(prob), maxiters=10000)
    prob = remake(prob, u0=res.minimizer)
    res = GalacticOptim.solve(prob, opt, cb = print_loss(prob), maxiters=200)
    phi = discretization.phi

    return PlasmaSolution(plasma, vars, dict_vars, phi, res, initθ, domains)
end

"""
Get the curl of a vector f w.r.t Ds
"""
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

"""
Print the loss of the loss function
"""
function print_loss(prob)
    pde_inner_loss_functions = prob.f.f.loss_function.pde_loss_function.pde_loss_functions.contents
    inner_loss_functions = prob.f.f.loss_function.bcs_loss_function.bc_loss_functions.contents

    cb = function (p,l)
        println("Current loss is: $l")
        println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
        println("bcs_losses: ", map(l_ -> l_(p), inner_loss_functions))
        return false
    end
    
    return cb
end
