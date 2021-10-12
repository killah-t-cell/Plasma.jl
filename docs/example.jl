# Example program
using Plasma

geometry(x,y,z) = x > 1 ? 0 : 1
plas = PlasmaParameters(temperature=30000, geometry=geometry, IC_e=maxwellian_3D, IC_i=maxwellian_3D)

@register plas.geometry(x,y,z)
@register plas.IC_e(vx,vy,vz,T,m, v_drift)
@register plas.IC_i(vx,vy,vz,T,m, v_drift)

solve_collisionless_plasma(plas, 0.0, 1.0; GPU=false)