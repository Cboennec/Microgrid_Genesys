#=
    Elec grid modelling
 =#
 """
 mutable struct Grid <: AbstractGrid

A mutable struct representing a Grid, which is a subtype of `AbstractGrid`.

# Parameters
- `powerMax::Float64`: The maximum power capacity of the grid.

# Variables
- `carrier::EnergyCarrier`: The energy carrier associated with the grid, such as electricity.
- `cost_in::AbstractArray{Float64,3}`: The input cost associated with the grid.
- `cost_out::AbstractArray{Float64,3}`: The output cost associated with the grid.
- `cost_exceed::AbstractArray{Float64,2}`: The cost associated with exceeding the grid's capacity.


## Example

```julia
grid = Grid(powerMax=36.0, carrier=Electricity())
```
"""
mutable struct Grid <: AbstractGrid
 # Parameters
 powerMax::Float64
 # Variables
 carrier::EnergyCarrier
 cost_in::AbstractArray{Float64,3}
 cost_out::AbstractArray{Float64,3}
 cost_exceed::AbstractArray{Float64,2}

 # Inner constructor
 Grid(; powerMax = 36., carrier = Electricity()) = new(powerMax, carrier)
end

### Preallocation
function preallocate!(grid::Grid, nh::Int64, ny::Int64, ns::Int64)
  grid.carrier.power = convert(SharedArray,zeros(nh, ny, ns))
  grid.cost_in = convert(SharedArray,zeros(nh, ny, ns))
  grid.cost_out = convert(SharedArray,zeros(nh, ny, ns))
  grid.cost_exceed = convert(SharedArray, ones(ny, ns) .* 10.2 ) # price from https://electricitedesavoie.fr/2017/05/09/3-points-comprendre-eviter-depassement-de-puissance-souscrite/#:~:text=Cette%20puissance%20est%20exprim%C3%A9e%20en,%2C11%E2%82%AC%2Fheure%20d%C3%A9pass%C3%A9e.
  return grid
end
