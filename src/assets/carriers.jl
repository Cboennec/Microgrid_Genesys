

"""
    Electricity

A mutable struct representing electricity as an energy carrier. It is a subtype of `EnergyCarrier`.

# Fields
- `power::AbstractArray{Float64,3}`: A 3-dimensional array of power values associated with electricity.

## Example 
```julia
Electricity()
```
"""
mutable struct Electricity <: EnergyCarrier
    power::AbstractArray{Float64,3}
    Electricity() = new()
end

"""
    Heat

A mutable struct representing heat as an energy carrier. It is a subtype of `EnergyCarrier`.

# Fields
- `power::AbstractArray{Float64,3}`: A 3-dimensional array of power values associated with heat.

## Example 
```julia
Heat()
```
"""
mutable struct Heat <: EnergyCarrier
    power::AbstractArray{Float64,3}
    Heat() = new()
end

"""
    Hydrogen

A mutable struct representing hydrogen as an energy carrier. It is a subtype of `EnergyCarrier`.

# Fields
- `power::AbstractArray{Float64,3}`: A 3-dimensional array of power values associated with hydrogen.

```julia
Hydrogen()
```
"""
mutable struct Hydrogen <: EnergyCarrier
    power::AbstractArray{Float64,3}
    Hydrogen() = new()
end
