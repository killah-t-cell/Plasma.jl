# Example program
using Plasma

geometry(x,y,z) = x > 1 ? 0 : 1
plas = PlasmaParameters(temperature=30000, geometry=geometry, IC_e=maxwellian_3D, IC_i=maxwellian_3D)

@register plas.geometry(x,y,z)
@register plas.IC_e(vx,vy,vz,T,m, v_drift)
@register plas.IC_i(vx,vy,vz,T,m, v_drift)

solve_collisionless_plasma(plas, 0.0, 1.0; GPU=false)

# Ideal interface
using Plasma

plasma = @collisionless_plasma begin
    species = begin
        d, 0.5 # concentration, mass, Z, charge
        d, 0.5
    end
    Tₑ, 100
    Tᵢ, 100
    nₑ, 100
    nᵢ, 200
end

initial_conditions = @plasma_ic begin

end

# what initial values
# what geometry
# what coils, walls, current

tspan = (0.,10.)
xspan = (0.,10.)
u0 = [fe_0, fi_0]

sol = solve_collisionless_plasma(plas, u0, tspan, xspan; GPU=false)
plot_plasma(sol)

# TODO this output should be better
plasma_params = PlasmaParameters(sol)
plasma_params.debye_length
# https://en.wikipedia.org/wiki/Plasma_parameters
# https://www.lanl.gov/projects/dense-plasma-theory/more/plasma-parameters.php
# I can access all values of plasma parameters computed at all points in time and Space