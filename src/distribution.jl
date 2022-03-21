# TODO add Kappa and other relevant distributions

"""
Temperature in eV
mass in Kg
"""
function Maxwellian(T, m; v_drift=zeros())
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

            1/(2*π*T) * exp(-(v_ - v_drift_)^2/(2*T))
        end
end