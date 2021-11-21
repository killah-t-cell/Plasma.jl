# TODO add Kappa and other relevant distributions

"""
Temperature in eV
mass in Kg
"""
function Maxwellian(T, m; v_drift=zeros())
        Kb = 8.617333262145e-5
        v_th = sqrt(2*Kb*T/m)

        function P(x,v) 
            if !(v isa Array)
                v = [v]    
            end

            if v_drift == zeros()
                v_drift = zeros(length(v))
            end

            if length(v) != length(v_drift)
                error("v and v_drift should have the same length")
            end

            v_ = sqrt(sum(v .^2))
            v_drift_ = sqrt(sum(v_drift.^2))

            (π*v_th^2)^(-3/2) * exp(-(v_ - v_drift_)^2/v_th^2)
        end
end