#=
    Loads modelling
 =#
 """
 Demand(; carrier = Electricity())

A mutable struct representing energy demand in the grid. The struct is a subtype of `AbstractDemand`.

# Fields
- `carrier::EnergyCarrier`: The type of energy carrier for the demand (e.g., Electricity, Heat, etc.) this structre also contains the quantity for each timestamp. see [`Main.Genesys.EnergyCarrier`](@ref)
- `timestamp::Array{DateTime,3}`: A 3-dimensional array of DateTime objects representing timestamps associated with the demand.

# Example
```julia
demand = Demand(carrier = Electricity())
"""
mutable struct Demand <: AbstractDemand
     # Variables
     carrier::EnergyCarrier
     timestamp::Array{DateTime,3}

     # Inner constructor
     Demand(; carrier = Electricity()) = new(carrier)
end

### Preallocation
function preallocate!(ld::Demand, nh::Int64, ny::Int64, ns::Int64)
    ld.carrier.power = convert(SharedArray,zeros(nh, ny, ns))
    ld.timestamp = Array{DateTime}(undef,(nh, ny, ns))
    return ld
end
