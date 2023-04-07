

"
    EnergyCarrier

Abstract type for energy carrier 

This is the generic container for energy carrier types
"
abstract type EnergyCarrier end

"
    Electricity()

Structure containing a 3D array of electricity powers (for each scenario for each year for each hour).

Should be incorporated into any structure implying carrying electricity.
"
mutable struct Electricity <: EnergyCarrier
    power::AbstractArray{Float64,3}
    Electricity() = new()
end

"
    Heat()

Structure containing a 3D array of electricity powers (for each scenario for each year for each hour).

Should be incorporated into any structure implying carrying heat.
"
mutable struct Heat <: EnergyCarrier
    power::AbstractArray{Float64,3}
    Heat() = new()
end

"
    Hydrogen()

Structure containing a 3D array of electricity powers (for each scenario for each year for each hour).

Should be incorporated into any structure implying carrying hydrogen.
"
mutable struct Hydrogen <: EnergyCarrier
    power::AbstractArray{Float64,3}
    Hydrogen() = new()
end
