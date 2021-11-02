abstract type AbstractDistribution end

@with_kw struct Maxwellian{Ty, V} <: AbstractDistribution
    v::V
    v_drift::V = zeros(length(v))
    T::Ty # Temperature in eV
    m::Ty # Mass in Kg
    P::Ty = nothing # probability distribution : 0 ≤ P ≤ 1

    function Maxwellian(v, T, m; P=nothing, v_drift=zeros(length(v)))
        if !(v isa Array)
            v = [v]    
        end

        if length(v) != length(v_drift)
            error("v and v_drift should have the same length")
        end

        v_ = sqrt(sum(v .^2))
        v_drift_ = sqrt(sum(v_drift.^2))
        Kb = 8.617333262145e-5
        v_th = sqrt(2*Kb*T/m)
        P = (π*v_th^2)^(-3/2) * exp(-(v_ - v_drift_)^2/v_th^2)

        new{typeof(m),typeof(v)}(
            v, v_drift, T, m, P
        )

    end
end