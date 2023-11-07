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



function get_η_E(P_net::Float64, fc::AbstractFuelCell)
  P_tot = floor(P_net / (1 - fc.k_aux); digits=6)
  
  #Find the corresponding current from an interpolation from P(I) curve 
  j = interpolation(fc.V_J[3,:], fc.V_J[1,:], P_tot, true )
  i = j * fc.surface

  return P_net / (fc.K * i * fc.N_cell)

end

#compute the power that correpond to the maximum allowed tension
function compute_min_power(fc::AbstractFuelCell)
  P_min_tot = interpolation(fc.V_J[2,:], fc.V_J[3,:], fc.V_max, false )
  P_min = P_min_tot / (1 + fc.k_aux)
  return P_min
end


function get_start_stops(powers::Vector{Float64})

  start_stop_count = 0
  # Count every start or stop
  for i in 1:(length(powers)-1)
      if (powers[i] > 0) != (powers[i+1] > 0)
          start_stop_count += 1
      end
  end

  #Return the number of start/stop cycle
  return start_stop_count/2
  
end