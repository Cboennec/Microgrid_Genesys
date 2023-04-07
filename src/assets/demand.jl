"
    Demand(; carrier = Electricity())

This structure models the load for a specific energy carrier.  

It consist of a power associated to a timestamp for each time-step of the simulation/optimization.
By default the energy carrier selected is Electricity
"
mutable struct Demand <: AbstractDemand
     # Variables
     carrier::EnergyCarrier
     timestamp::Array{DateTime,3}

     # Inner constructor
     Demand(; carrier = Electricity()) = new(carrier)
end

"
    preallocate!(ld::Demand, nh::Int64, ny::Int64, ns::Int64)

Create the data structure store powers and their associated time stamp.

#TODO : why is their a return ?
"
function preallocate!(ld::Demand, nh::Int64, ny::Int64, ns::Int64)
    ld.carrier.power = convert(SharedArray,zeros(nh, ny, ns))
    ld.timestamp = Array{DateTime}(undef,(nh, ny, ns))
    return ld
end
