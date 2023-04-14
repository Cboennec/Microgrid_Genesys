#=
    Heater modelling
 =#
 """
 mutable struct Heater <: AbstractConverter

A mutable struct representing a Heater, which is a subtype of `AbstractConverter`.

# Parameters
- `η_E_H::Float64`: The conversion efficiency of the heater.
- `lifetime::Int64`: The expected lifetime of the heater (in years).
- `bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}`: The lower and upper bounds of the heater's power.
- `powerMax_ini::Float64`: The initial maximum power output of the heater.
- `soh_ini::Float64`: The initial state of health of the heater.

# Variables
- `powerMax::AbstractArray{Float64,2}`: The maximum power output of the heater.
- `carrier::Vector{Any}`: The carrier vector.
- `cost::AbstractArray{Float64,2}`: The cost associated with the heater.


## Example

```julia
heater = Heater(η_E_H=1.0, lifetime=25, bounds=(lb=30.0, ub=30.0), powerMax_ini=30.0, soh_ini=1.0)
```
"""
mutable struct Heater <: AbstractConverter
  # Parameters
  η_E_H::Float64
  lifetime::Int64
  bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
  # Initial conditions
  powerMax_ini::Float64
  soh_ini::Float64
  # Variables
  powerMax::AbstractArray{Float64,2}
  carrier::Vector{Any}
  # Eco
  cost::AbstractArray{Float64,2}
  # Inner constructor
  Heater(; η_E_H = 1.,
          lifetime = 25,
          bounds = (lb = 30., ub = 30.),
          powerMax_ini = 30.,
          soh_ini = 1.) =
          new(η_E_H, lifetime, bounds, powerMax_ini)
end

### Preallocation
function preallocate!(heater::Heater, nh::Int64, ny::Int64, ns::Int64)
  heater.powerMax = convert(SharedArray,zeros(ny+1, ns)) ; heater.powerMax[1,:] .= heater.powerMax_ini
  heater.carrier = [Electricity(), Heat()]
  heater.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
  heater.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
  heater.cost = convert(SharedArray,zeros(ny, ns))
  return heater
end

### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, heater::Heater, decision::Float64, Δh::Int64)
 heater.carrier[1].power[h,y,s], heater.carrier[2].power[h,y,s] = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), decision, Δh)
end

function compute_operation_dynamics(heater::Heater, state::NamedTuple{(:powerMax,), Tuple{Float64}}, decision::Float64, Δh::Int64)
 # Power constraint and correction
 power_E = min(max(decision, -state.powerMax), 0.)
 # Power computation
 power_H = - heater.η_E_H * power_E
 return power_E, power_H
end

### Investment dynamic
function compute_investment_dynamics!(y::Int64, s::Int64, heater::Heater, decision::Union{Float64, Int64})
 heater.powerMax[y+1,s] = compute_investment_dynamics(heater, (powerMax = heater.powerMax[y,s],), decision)
end

function compute_investment_dynamics(heater::Heater, state::NamedTuple{(:powerMax,), Tuple{Float64}}, decision::Union{Float64, Int64})
 if decision > 1e-2
     powerMax_next = decision
 else
     powerMax_next = state.powerMax
 end
 return powerMax_next
end
