#=
    Sources modelling
 =#
 """
 mutable struct Solar <: AbstractGeneration

A mutable struct representing a solar power generation source. It contains information about the lifetime, bounds, initial conditions, variables, and costs related to the solar generation.

# Parameters
-`lifetime::Int64 (default: 25)`: The lifetime of the solar power generation system.
-`bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}} (default: (lb = 0., ub = 1000.))`: The lower and upper bounds of the solar power generation system.
-`powerMax_ini::Float64 (default: 0.)`: The initial maximum power output of the solar power generation system.
-`soh_ini::Float64 (default: 1.)`: The initial state of health of the solar power generation system.
"""
mutable struct Solar <: AbstractGeneration
  lifetime::Int64
  bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
  # Initial conditions
  powerMax_ini::Float64
  soh_ini::Float64
  # Variables
  carrier::Electricity
  powerMax::AbstractArray{Float64,2}
  timestamp::Array{DateTime,3}
  # Eco
  cost::AbstractArray{Float64,2}
  # Inner constructor
  Solar(;lifetime=25, bounds = (lb = 0., ub = 1000.), powerMax_ini = 0., soh_ini = 1.) = new(lifetime, bounds, powerMax_ini, soh_ini)
end

"""
 preallocate!(pv::Solar, nh::Int64, ny::Int64, ns::Int64)

Preallocate necessary arrays for the solar power generation model with the specified number of hours (nh), years (ny), and scenarios (ns).
"""
function preallocate!(pv::Solar, nh::Int64, ny::Int64, ns::Int64)
 pv.carrier = Electricity()
 pv.carrier.power = convert(SharedArray,zeros(nh, ny, ns))
 pv.powerMax = convert(SharedArray,zeros(ny+1, ns)) ; pv.powerMax[1,:] .= pv.powerMax_ini
 pv.timestamp = Array{DateTime}(undef,(nh, ny, ns))
 pv.cost = convert(SharedArray,zeros(ny, ns))

 return pv
end

"""
 compute_investment_dynamics!(y::Int64, s::Int64, pv::Solar, decision::Union{Float64, Int64})

Compute the investment dynamics of the solar power generation system for a given year (y) and scenario (s), using the provided decision.
"""
function compute_investment_dynamics!(y::Int64, s::Int64, pv::Solar, decision::Union{Float64, Int64})
  pv.powerMax[y+1,s] = compute_investment_dynamics(pv, (powerMax = pv.powerMax[y,s],), decision)
end



function initialize_investments!(s::Int64, pv::Solar, decision::Union{Float64, Int64})
  pv.powerMax[1,s] = decision
end


"""
compute_investment_dynamics(pv::Solar, state::NamedTuple{(:powerMax,), Tuple{Float64}}, decision::Union{Float64, Int64})

Compute the investment dynamics of the solar power generation system for the given state and decision.
"""
function compute_investment_dynamics(pv::Solar, state::NamedTuple{(:powerMax,), Tuple{Float64}}, decision::Union{Float64, Int64})
  if decision > 1e-2
      powerMax_next = decision
  else
      powerMax_next = state.powerMax
  end
  return powerMax_next
end
