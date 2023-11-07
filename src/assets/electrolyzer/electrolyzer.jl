abstract type AbstractElectrolyzer <: AbstractConverter  end


#=
    Electrolyzer modelling
 =#
 """
 mutable struct Electrolyzer <: AbstractConverter

A mutable struct representing an Electrolyzer, which is a subtype of `AbstractConverter`.

# Parameters
- `α_p::Float64`: The performance degradation factor.
- `η_E_H2::Float64`: The conversion efficiency of electricity to hydrogen.
- `η_E_H::Float64`: The conversion efficiency of electricity to heat.
- `lifetime::Int64`: The expected lifetime of the electrolyzer (in years).
- `nHoursMax::Float64`: The maximum number of operational hours for the electrolyzer.
- `bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}`: The lower and upper bounds of the electrolyzer's power.

# Initial conditions
- `powerMax_ini::Float64`: The initial maximum power output of the electrolyzer.
- `soh_ini::Float64`: The initial state of health of the electrolyzer.

# Variables
- `powerMax::AbstractArray{Float64,2}`: The maximum power output of the electrolyzer.
- `carrier::Vector{EnergyCarrier}`: The vector of energy carriers associated with the electrolyzer.
- `soh::AbstractArray{Float64,3}`: The state of health of the electrolyzer.

# Eco
- `cost::AbstractArray{Float64,2}`: The cost associated with the electrolyzer.


## Example

```julia
electrolyzer = Electrolyzer(α_p=0.05, η_E_H2=0.5, η_E_H=0.3, lifetime=15, nHoursMax=26000.0, bounds=(lb=0.0, ub=50.0), powerMax_ini=0.0, soh_ini=1.0)
```
"""
mutable struct Electrolyzer <: AbstractElectrolyzer
  # Paramètres
  α_p::Float64
  η_E_H2::Float64
  η_E_H::Float64
  lifetime::Int64
  nHoursMax::Float64
  bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
  # Initial conditions
  powerMax_ini::Float64
  soh_ini::Float64
  # Variables
  powerMax::AbstractArray{Float64,2}
  carrier::Vector{EnergyCarrier}
  soh::AbstractArray{Float64,3}
  # Eco
  cost::AbstractArray{Float64,2}
  # Inner constructor
  Electrolyzer(; α_p = 5/100,
              η_E_H2 = 0.5,
              η_E_H = 0.3,
              lifetime = 15,
              nHoursMax = 26000.,
              bounds = (lb = 0., ub = 50.),
              powerMax_ini = 0.,
              soh_ini = 1.) =
              new(α_p, η_E_H2, η_E_H, lifetime, nHoursMax, bounds, powerMax_ini, soh_ini)
end

### Preallocation
function preallocate!(elyz::Electrolyzer, nh::Int64, ny::Int64, ns::Int64)
  elyz.powerMax = convert(SharedArray,zeros(ny+1, ns)) ; elyz.powerMax[1,:] .= elyz.powerMax_ini
  elyz.carrier = [Electricity(), Heat(), Hydrogen()]
  elyz.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
  elyz.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
  elyz.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
  elyz.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; elyz.soh[1,1,:] .= elyz.soh_ini
  elyz.cost = convert(SharedArray,zeros(ny, ns))
  return elyz
end

### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, elyz::Electrolyzer, decision::Float64, Δh::Int64)
 elyz.carrier[1].power[h,y,s], elyz.carrier[2].power[h,y,s], elyz.carrier[3].power[h,y,s] =
 compute_operation_dynamics(elyz, h, y, s, decision, Δh)
end

function compute_operation_dynamics(elyz::Electrolyzer, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)
 # Power constraint and correction
 elyz.α_p * elyz.powerMax[y,s] >= decision && elyz.soh[h,y,s] * elyz.nHoursMax / Δh > 1. ? power_E = max(decision, -elyz.powerMax[y,s]) : power_E = 0.
 # Power conversion
 power_H2 = - power_E * elyz.η_E_H2
 power_H = - power_E * elyz.η_E_H
 # SoH computation
 elyz.soh[h+1,y,s] = elyz.soh[h,y,s] - (power_E > 0.) * Δh / elyz.nHoursMax
 return power_E, power_H, power_H2
end

### Investment dynamic
function compute_investment_dynamics!(y::Int64, s::Int64, elyz::Electrolyzer, decision::Union{Float64, Int64})
 elyz.powerMax[y+1,s], elyz.soh[1,y+1,s] = compute_investment_dynamics(elyz, (powerMax = elyz.powerMax[y,s], soh = elyz.soh[end,y,s]), decision)
end


function initialize_investments!(s::Int64, elyz::Electrolyzer, decision::Union{Float64, Int64})
	elyz.powerMax[1,s] = decision
	elyz.soh[1,1,s] = elyz.soh_ini
end


function compute_investment_dynamics(elyz::Electrolyzer, state::NamedTuple{(:powerMax, :soh), Tuple{Float64, Float64}}, decision::Union{Float64, Int64})
 if decision > 1e-2
     powerMax_next = decision
     soh_next = 1.
 else
     powerMax_next = state.powerMax
     soh_next = state.soh
 end
 return powerMax_next, soh_next
end



#compute the power that correpond to the maximum allowed tension
function compute_min_power(elyz::AbstractElectrolyzer)
  P_min = interpolation(elyz.V_J[1,:], elyz.V_J[3,:], elyz.J_min, true )
  P_min_tot = P_min * (1 + elyz.k_aux)
  return P_min_tot
end


function get_η_E(P_brut::Float64, elyz::AbstractElectrolyzer)
  P_net = ceil(P_brut / (1 + elyz.k_aux); digits=6)
  
  #Find the corresponding current from an interpolation from P(I) curve 
  j = interpolation(elyz.V_J[3,:], elyz.V_J[1,:], P_net, true)
  i = j * elyz.surface

  return elyz.K * i / (P_brut / elyz.N_cell)

end