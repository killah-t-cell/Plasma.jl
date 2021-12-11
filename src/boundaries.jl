"""
Set Neumann boundary conditions
"""
function Neumann(t, xs, Ds, Fs, bc; bc_value = 0)
    bcs = []
    for F in Fs
        for i in eachindex(xs)
            temp = xs[i]
            xs[i] = bc
            push!(bcs,Ds[i](F(t, xs...)) ~ bc_value)
            xs[i] = temp
        end    
    end
    
    return bcs
end

"""
Set Dirichlet boundary conditions

Dirichlet(t, xs, Es, lb, bc_value=2)

where Es must be the same length as xs.
"""
function Dirichlet(t, xs, Ds, Fs, bc; bc_value = 0)
    bcs = []
    for F in Fs
        for i in eachindex(xs)
            temp = xs[i]
            xs[i] = bc
            push!(bcs,F(t, xs...) ~ bc_value)
            xs[i] = temp
        end    
    end
    
    return bcs
end

"""
Set reflective boundary conditions (for f)
"""
function Reflective(t, xs, vs, Fs, bc, a, g)
    bcs = []
    for F in Fs
        for i in eachindex(vs)
            temp = vs[i]
            vs[i] = bc
            lhs_value = F(t, xs..., vs...)
            vs[i] = -bc
            rhs_value = F(t, xs..., vs...)
            push!(bcs,lhs_value ~ a * rhs_value + g)
            vs[i] = temp
        end    
    end
    
    return bcs
end

