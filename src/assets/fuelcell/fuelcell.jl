abstract type AbstractFuelCell <: AbstractConverter  end


function compute_powers(fc::AbstractFuelCell, state::NamedTuple{(:powerMax, :soh), Tuple{Float64, Float64}}, decision::Float64, Δh::Int64)
  fc.α_p * state.powerMax <= decision && state.soh * fc.nHoursMax / Δh > 1. ? power_E = min(decision, state.powerMax) : power_E = 0.
  # Power conversion
  power_H2 = - power_E / fc.η_H2_E
  power_H = - power_H2 * fc.η_H2_H
  # SoH computation
  return power_E, power_H, power_H2
end

function count_OnOff(power_E::Vector{Float64})

  binary_On = power_E .> 0

  result_OnOff = zeros(length(power_E))

  for i in 2:(length(binary_On))
      result_OnOff[i] = binary_On[i] - binary_On[i-1] # vector of On Offs 1 means an activation -1 means a stop
  end
  
  return result_OnOff
  
end   



