abstract type AbstractHeater <: AbstractConverter  end

abstract type AbstractHeaterEffModel end

abstract type AbstractHeaterAgingModel end



#=
    Heater modelling
 =#
 """
 mutable struct Heater <: AbstractConverter

A mutable struct representing a Heater, which is a subtype of `AbstractConverter`.

# Parameters
- `eff_model::AbstractHeaterEffModel`: The model for conversion efficiency of the heater.
- `SoH_model::AbstractHeaterAgingModel`: The aging model for the heater. The default model use a fixed expected lifetime for the heater (in years).
- `bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}`: The lower and upper bounds of the heater's power.
- `powerMax_ini::Float64`: The initial maximum power output of the heater.
- `soh_ini::Float64`: The initial state of health of the heater.

# Variables
- `powerMax::AbstractArray{Float64,2}`: The maximum power output of the heater.
- `carrier::Vector{Any}`: The carrier vector.
- `cost::AbstractArray{Float64,2}`: The cost associated with the heater.


## Example

```julia
heater = Heater(eff_model=ConstEfficiencyHeater(η_E_H=1.0), SoH_model=FixedLifetimeHeater(lifetime=25), bounds=(lb=30.0, ub=30.0), powerMax_ini=30.0, soh_ini=1.0)
```
"""
mutable struct Heater <: AbstractHeater
  # Parameters
  eff_model::AbstractHeaterEffModel
  SoH_model::AbstractHeaterAgingModel
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
  Heater(; eff_model = ConstEfficiencyHeater(),
          SoH_model = FixedLifetimeHeater(),
          bounds = (lb = 30., ub = 30.),
          powerMax_ini = 30.,
          soh_ini = 1.) =
          new(eff_model, SoH_model, bounds, powerMax_ini)
end

mutable struct ConstEfficiencyHeater <: AbstractHeaterEffModel

  η_E_H::Float64

  ConstEfficiencyHeater(;η_E_H = 1.) = new(η_E_H)
end


mutable struct FixedLifetimeHeater <: AbstractHeaterAgingModel

  lifetime::Int64

  FixedLifetimeHeater(;lifetime = 25) = new(lifetime)
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
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, heater::Heater, decision::Float64, Δh::Float64)
 heater.carrier[1].power[h,y,s], heater.carrier[2].power[h,y,s] = compute_operation_dynamics(heater, (powerMax = heater.powerMax[y,s],), decision, Δh)
end

function compute_operation_dynamics(heater::Heater, state::NamedTuple{(:powerMax,), Tuple{Float64}}, decision::Float64, Δh::Float64)
 # Power constraint and correction
 power_E = min(max(decision, -state.powerMax), 0.)
 # Power computation
 power_H = - heater.eff_model.η_E_H * power_E
 return power_E, power_H
end

### Investment dynamic
function compute_investment_dynamics!(y::Int64, s::Int64, heater::Heater, decision::Union{Float64, Int64})
 heater.powerMax[y+1,s] = compute_investment_dynamics(heater, (powerMax = heater.powerMax[y,s],), decision)
end



function initialize_investments!(s::Int64, heater::Heater, decision::Union{Float64, Int64})
  heater.powerMax[1,s] = decision
end



function compute_investment_dynamics(heater::Heater, state::NamedTuple{(:powerMax,), Tuple{Float64}}, decision::Union{Float64, Int64})
 if decision > 1e-2
     powerMax_next = decision
 else
     powerMax_next = state.powerMax
 end
 return powerMax_next
end
