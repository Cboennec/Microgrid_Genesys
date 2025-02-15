abstract type AbstractH2TankEffModel end

abstract type AbstractH2TankAgingModel end

mutable struct FixedH2TankEfficiency <: AbstractH2TankEffModel

	η_ch::Float64 #Charging efficiency
	η_dch::Float64 #Discharging efficiency
	α_p_ch::Float64 #C_rate max
	α_p_dch::Float64 #C_rate max
	η_self::Float64 #Auto discarge factor
	
	
	FixedH2TankEfficiency(;η_ch = 1.,
		η_dch = 1.,
		α_p_ch = 1.5,
		α_p_dch = 1.5,
		η_self = 0.0,
		) = new(η_ch, η_dch, α_p_ch, α_p_dch, η_self)

end

mutable struct FixedH2TankLifetime <: AbstractH2TankAgingModel

	lifetime::Int64

	FixedH2TankLifetime(;lifetime = 25) = new(lifetime)
end


 """
 mutable struct H2Tank  <: AbstractStorage

A mutable struct representing a hydrogen tank storage model with various parameters, initial conditions, variables, and an inner constructor.

# Parameters
- `α_p_ch::Float64`: Maximum charging power factor (default: 1.5)
- `α_p_dch::Float64`: Maximum discharging power factor (default: 1.5)
- `η_ch::Float64`: Charging efficiency (default: 1.0)
- `η_dch::Float64`: Discharging efficiency (default: 1.0)
- `η_self::Float64`: Self-discharge rate (default: 0.0)
- `α_soc_min::Float64`: Minimum state of charge factor (default: 0.0)
- `α_soc_max::Float64`: Maximum state of charge factor (default: 1.0)
- `lifetime::Int64`: Storage lifetime in years (default: 25)
- `bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}`: Lower and upper bounds of storage capacity (default: (lb = 0., ub = 10000.))
- `Erated_ini::Float64`: Initial rated storage energy capacity (default: 1e-6)
- `soc_ini::Float64`: Initial state of charge (default: 0.5)
- `soh_ini::Float64`: Initial state of health (default: 1.0)
- `Erated::AbstractArray{Float64,2}`: Rated storage energy capacity over time and scenarios
- `carrier::Hydrogen`: Hydrogen carrier for the H2 tank
- `soc::AbstractArray{Float64,3}`: State of charge over time, years, and scenarios
- `cost::AbstractArray{Float64,2}`: Cost of the hydrogen storage over time and scenarios
"""
mutable struct H2Tank  <: AbstractStorage
   # Paramètres
   pression_max::Float64 # Pression en Bar
   eff_model::AbstractH2TankEffModel # Efficiency model
   α_soc_min::Float64
   α_soc_max::Float64
   SoH_model::AbstractH2TankAgingModel # Aging model
   bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
   # Initial conditions
   Erated_ini::Float64
   soc_ini::Float64
   soh_ini::Float64
   # Variable
   Erated::AbstractArray{Float64,2}
   carrier::Hydrogen
   soc::AbstractArray{Float64,3}
   # Eco
   cost::AbstractArray{Float64,2}
   # Inner constructor
   H2Tank(;
      pression_max = 150., # Pression en Bar,
      eff_model = FixedH2TankEfficiency(),
      α_soc_min = .05 ,
      α_soc_max = 1.,
      SoH_model = FixedH2TankLifetime(),
      bounds = (lb = 0., ub = 10000.),
      Erated_ini = 1e-6,
      soc_ini = 0.5,
      soh_ini = 1.) =
      new(pression_max, eff_model, α_soc_min, α_soc_max, SoH_model, bounds, Erated_ini, soc_ini, soh_ini)
end

"""
  preallocate!(h2tank::H2Tank, nh::Int64, ny::Int64, ns::Int64)

Preallocate memory for the `H2Tank` object with given dimensions for hours (nh), years (ny), and scenarios (ns).
"""
function preallocate!(h2tank::H2Tank, nh::Int64, ny::Int64, ns::Int64)
  h2tank.Erated = convert(SharedArray,zeros(ny+1, ns)) ; h2tank.Erated[1,:] .= h2tank.Erated_ini
  h2tank.carrier = Hydrogen()
  h2tank.carrier.power = convert(SharedArray,zeros(nh, ny, ns))
  h2tank.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; h2tank.soc[1,1,:] .= h2tank.soc_ini
  h2tank.cost = convert(SharedArray,zeros(ny, ns))
  return h2tank
end

"""
  compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, h2tank::H2Tank, decision::Float64, Δh::Float64)

Compute the operation dynamics of the hydrogen tank storage for a given hour (h), year (y), and scenario (s), using the provided decision and time step (Δh).
"""
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, h2tank::H2Tank, decision::Float64, Δh::Float64)
   h2tank.soc[h+1,y,s], h2tank.carrier.power[h,y,s] = compute_operation_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[h,y,s]), decision, Δh)
end

"""
  compute_operation_dynamics(h2tank::H2Tank, state::NamedTuple{(:Erated, :soc), Tuple{Float64, Float64}}, decision::Float64, Δh::Float64)

Compute the operation dynamics of the hydrogen tank storage for the given state, decision, and time step (Δh).
"""
function compute_operation_dynamics(h2tank::H2Tank, state::NamedTuple{(:Erated, :soc), Tuple{Float64, Float64}}, decision::Float64, Δh::Float64)
  # Control power constraint and correction
  power_dch = max(min(decision, h2tank.eff_model.α_p_dch * state.Erated, h2tank.eff_model.η_dch * (state.soc * (1. - h2tank.eff_model.η_self * Δh) - h2tank.α_soc_min) * state.Erated / Δh), 0.)
  power_ch = min(max(decision, -h2tank.eff_model.α_p_ch * state.Erated, (state.soc * (1. - h2tank.eff_model.η_self * Δh) - h2tank.α_soc_max) * state.Erated / Δh / h2tank.eff_model.η_ch), 0.)
  # SoC dynamic
  if state.Erated != 0
    soc_next = state.soc * (1. - h2tank.eff_model.η_self * Δh) - (power_ch * h2tank.eff_model.η_ch + power_dch / h2tank.eff_model.η_dch) * Δh / state.Erated
  else
    soc_next = 0
  end
  
  return soc_next, power_dch + power_ch
end

"""
  compute_investment_dynamics!(y::Int64, s::Int64, h2tank::H2Tank, decision::Union{Float64, Int64})

Compute the investment dynamics of the hydrogen tank storage for a given year (y) and scenario (s), using the provided decision.
"""
function compute_investment_dynamics!(y::Int64, s::Int64, h2tank::H2Tank, decision::Union{Float64, Int64})
  h2tank.Erated[y+1,s], h2tank.soc[1,y+1,s] = compute_investment_dynamics(h2tank, (Erated = h2tank.Erated[y,s], soc = h2tank.soc[end,y,s]), decision)
end


function initialize_investments!(s::Int64, h2tank::H2Tank, decision::Union{Float64, Int64})
	h2tank.Erated[1,s] = decision
	h2tank.soc[1,1,s] = h2tank.soc_ini
end

"""
compute_investment_dynamics(h2tank::H2Tank, state::NamedTuple{(:Erated, :soc), Tuple{Float64, Float64}}, decision::Union{Float64, Int64})

Compute the investment dynamics of the hydrogen tank storage for the given state and decision.
"""
function compute_investment_dynamics(h2tank::H2Tank, state::NamedTuple{(:Erated, :soc), Tuple{Float64, Float64}}, decision::Union{Float64, Int64})
  if decision > 1e-2
      Erated_next = decision
      soc_next = h2tank.soc_ini
  else
      Erated_next = state.Erated
      soc_next = state.soc
  end
  return Erated_next, soc_next
end
