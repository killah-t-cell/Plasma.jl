# TODO add ToroidalGeometry, LinearGeometry and other structs that let you pick parameters (like Toroidal parameters to define a plasma)
@with_kw struct Geometry{F <: Function} <: AbstractGeometry
    f::F = (_) -> 1 # defining function
end