abstract type AbstractFuelCell <: AbstractConverter  end

abstract type AbstractFuelCellEffModel end

abstract type AbstractFuelCellAgingModel end





mutable struct FixedFuelCellEfficiency <: AbstractFuelCellEffModel
 
  α_p::Float64 #Minimum power defined as a share of the maximum Power
  η_H2_E::Float64 #The efficiency from DiHydrogen to Electricity
  η_H2_H::Float64 #The efficiency from DiHydrogen to Heat
  k_aux::Float64
  powerMax_ini::Float64
  couplage::Bool 

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell
  powerMin::AbstractArray{Float64,3} #The minimum power that can be demanded to the FuelCell

  V_J::AbstractArray{Float64,2}


  FixedFuelCellEfficiency(;α_p = 0.08,
    η_H2_E = 0.4,
    η_H2_H = 0.4,
    k_aux = 0.15,
    powerMax_ini =.00001,
    couplage = false
    ) = new(α_p, η_H2_E, η_H2_H, k_aux, powerMax_ini, couplage)
end

mutable struct PolarizationFuelCellEfficiency <: AbstractFuelCellEffModel
  k_aux::Float64 # Share of the power used by the auxiliaries
  couplage::Bool
  K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
  powerMax_ini::Float64
  V_max::Float64

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell
  powerMin::AbstractArray{Float64,3} #The minimum power that can be demanded to the FuelCell

  V_J::AbstractArray{Float64,2}



  PolarizationFuelCellEfficiency(; k_aux = 0.15,
  couplage = true,
  K = (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321),  #PCI * M_H2 * λ * 3600/(2*F)
  powerMax_ini = .00001,
  V_max = 0.8
  ) = new(k_aux, couplage, K, powerMax_ini, V_max)
end


mutable struct LinearFuelCellEfficiency <: AbstractFuelCellEffModel
  k_aux::Float64 # Share of the power used by the auxiliaries
  couplage::Bool
  K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
  powerMax_ini::Float64
  V_max::Float64

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell
  powerMin::AbstractArray{Float64,3} #The minimum power that can be demanded to the FuelCell

  a_η::Float64 # the slope for the fucntion η(P)
  b_η::Float64 # the ordinate at the origin for the function η(P)

  V_J::AbstractArray{Float64,2}


  LinearFuelCellEfficiency(; k_aux = 0.15,
  couplage = true,
  K = (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321),  #PCI * M_H2 * λ * 3600/(2*F)
  powerMax_ini = .00001,
  V_max = 0.8
  ) = new(k_aux, couplage, K, powerMax_ini, V_max)
end



mutable struct deg_params
  a_slope::Float64
  b_slope::Float64
  c_slope::Float64
  power_slope::Float64
  b::Float64
  adjustment_coef::Float64
  start_stop_coef::Float64
end

#M/A en option
mutable struct PowerAgingFuelCell <: AbstractFuelCellAgingModel
  plot::Bool 
  StartStop::Bool
  deg_params::deg_params 
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current

  V_J_ini::AbstractArray{Float64,2}
  V_J::AbstractArray{Float64,2}

  PowerAgingFuelCell(;plot = false,
    StartStop = true,
    deg_params = nothing,
    update_by_year = 12,
    J_ref = 0.62
  ) = new(plot, StartStop, deg_params, update_by_year, J_ref)
end



#M/A en option
mutable struct FunctHoursAgingFuelCell <: AbstractFuelCellAgingModel
  plot::Bool 
  StartStop::Bool
  deg_params::deg_params 
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current density
  J_base::Float64 # The current density used for degradation

  V_J_ini::AbstractArray{Float64,2}
  V_J::AbstractArray{Float64,2}

  coef_a::Float64 #The slope of voltage degradation for each functioning hour
  coef_b::Float64 #The ordinate at origin of voltage degradation for each functioning hour

  FunctHoursAgingFuelCell(;plot = false,
    StartStop = true,
    deg_params = nothing,
    update_by_year = 12,
    J_ref = 0.62,
    J_base = 0.1
  ) = new(plot, StartStop, deg_params, update_by_year, J_ref, J_base)
end



#M/A en option
mutable struct FixedLifetimeFuelCell <: AbstractFuelCellAgingModel
  plot::Bool 
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current density
  nHourMax::Int64

  V_J_ini::AbstractArray{Float64,2}
  V_J::AbstractArray{Float64,2}

  V_nom_ini::Float64

  FixedLifetimeFuelCell(;plot = false,
    update_by_year = 12,
    J_ref = 0.62,
    nHourMax = 87600
  ) = new(plot, update_by_year, J_ref, nHourMax)
end




mutable struct FuelCell <: AbstractFuelCell

  SoC_model::AbstractFuelCellEffModel
	SoH_model::AbstractFuelCellAgingModel
	
	bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
	SoH_threshold::Float64 # SoH level to replace battery
	couplage::Bool  #a boolean tuple to tell wether or not the soh should influence the other parameters.
  V_J_ini::AbstractArray{Float64,2}

	# Initial conditions
	soc_ini::Float64 # first state of charge for the begining of simulation
	soh_ini::Float64 # first state of health for the begining of simulation

  N_cell::Int64 #The number of assembled cells
  surface::Float64 #The number of assembled cells



  η::AbstractArray{Float64,3}
  carrier::Vector{EnergyCarrier}
  soh::AbstractArray{Float64,3}

  # Eco
  cost::AbstractArray{Float64,2}

	FuelCell(;SoC_model = PolarizationFuelCellEfficiency(),
    SoH_model = PowerAgingFuelCell(),
    bounds = (lb = 0., ub = 50.),
    SoH_threshold = 0.9,
    couplage = true,
    V_J_ini = nothing,
    soc_ini = 0.5,
    soh_ini = 1. 
  ) = new(SoC_model, SoH_model, bounds, SoH_threshold, couplage, V_J_ini, soc_ini, soh_ini)

end



  
  ### Preallocation
  function preallocate!(fc::FuelCell, nh::Int64, ny::Int64, ns::Int64)

    fc.SoC_model.powerMax = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  fc.SoC_model.powerMax[1,1,:] .= fc.SoC_model.powerMax_ini
    fc.SoC_model.powerMin = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  fc.SoC_model.powerMin[1,1,:] .= fc.SoC_model.powerMax_ini
    fc.η = convert(SharedArray,zeros(nh+1, ny+1, ns))
    fc.carrier = [Electricity(), Heat(), Hydrogen()]
    fc.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; fc.soh[1,1,:] .= fc.soh_ini
    fc.cost = convert(SharedArray,zeros(ny, ns))

    fc.SoH_model.V_J_ini = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P
    fc.SoH_model.V_J = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P
    fc.SoC_model.V_J = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P

    return fc
end
  

### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, fc::FuelCell, decision::Float64, Δh::Int64)

  fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] = compute_operation_soc(fc, fc.SoC_model, h ,y ,s , decision, Δh)

  fc.soh[h+1,y,s], fc.SoC_model.powerMax[h+1,y,s], fc.SoC_model.powerMin[h+1,y,s] = compute_operation_soh(fc, fc.SoH_model, h ,y ,s, Δh)

end


### Operation dynamic
function compute_operation_dynamics(fc::FuelCell, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)

  return compute_operation_soc(fc, fc.SoC_model, h ,y ,s , decision, Δh)

end


function compute_operation_soc(fc::FuelCell, model::PolarizationFuelCellEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
 #Apply minimum power
  model.powerMin[h,y,s] <= decision ? power_E = min(decision, model.powerMax[h,y,s]) : power_E = 0. 
      
  if power_E > 0
      #Compute the power needed to feed the auxiliaries and deliver the right power
      P_tot = floor(power_E / (1 - model.k_aux); digits=6)

      #Find the corresponding current from an interpolation from P(I) curve 
      j = interpolation(model.V_J[3,:], model.V_J[1, :], P_tot, true)
     
      i = j * fc.surface
      
      η_E = power_E / (model.K * i * fc.N_cell)

      fc.η[h,y,s] = η_E

      η_H = 0.8 - η_E

      elec_power, heat_power, hydrogen_power = (power_E), (power_E * η_H / η_E),  - (power_E / η_E) 
  
  else 
      
    elec_power, heat_power, hydrogen_power = 0., 0., 0.
  end

  return elec_power, heat_power, hydrogen_power 
end



function compute_operation_soc(fc::FuelCell, model::FixedFuelCellEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
  #Apply minimum power
   model.powerMin[h,y,s] <= decision ? power_E = min(decision, model.powerMax[h,y,s]) : power_E = 0. 
       
   if power_E > 0
       elec_power, heat_power, hydrogen_power = (power_E), (power_E * model.η_H2_H / model.η_H2_E),  - (power_E / model.η_H2_E)  
       fc.η[h,y,s] = elec_power / hydrogen_power
   else 
       
     elec_power, heat_power, hydrogen_power = 0., 0., 0.
   end
 
   return elec_power, heat_power, hydrogen_power 
 end


 
function compute_operation_soc(fc::FuelCell, model::LinearFuelCellEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
  #Apply minimum power
   model.powerMin[h,y,s] <= decision ? power_E = min(decision, model.powerMax[h,y,s]) : power_E = 0. 
       
   if power_E > 0
      #Compute the power needed to feed the auxiliaries and deliver the right power
      P_tot = floor(power_E / (1 - model.k_aux); digits=6)
 
      η_E = model.a_η * P_tot + model.b_η 
      
      if η_E >= 0.45 && y==9 && h < 1000
        println("y,h = ", y, ", ", h, "   a_η = ", model.a_η, " , P_tot = ", P_tot, ", b_η = ",  model.b_η )
      end
      #println("y,h = ", y, ", ", h, "a_η = ", model.a_η, " , P_tot = ", P_tot, ", b_η = ",  model.b_η )


      fc.η[h,y,s] = η_E

      η_H = 0.8 - η_E

      elec_power, heat_power, hydrogen_power = (power_E), (power_E * η_H / η_E),  - (power_E / η_E) 
   
   else 

      elec_power, heat_power, hydrogen_power = 0., 0., 0.
   end
 
   return elec_power, heat_power, hydrogen_power 
end
 


 

function compute_operation_soh(fc::FuelCell, model::PowerAgingFuelCell, h::Int64,  y::Int64,  s::Int64,  Δh::Int64)

  if (h%convert(Int64,floor(8760/model.update_by_year))) == 0 
    interval = (h-convert(Int64,floor(8760/model.update_by_year))+1):h

    powers = fc.carrier[1].power[interval,y,s]

    coef_b = model.deg_params.b

    current_densities = []
    
    # get the sequence of intensities
    for p in powers
        if p .> 1e-6
            push!(current_densities, interpolation(model.V_J[3,:], model.V_J[1,:], p, true))
        end
    end

    for j in current_densities
       coef_a = get_slope_deg(j, model.deg_params.power_slope, model.deg_params.a_slope, model.deg_params.b_slope, model.deg_params.c_slope)
 
       ΔV = (model.V_J[1,:] * coef_a).+coef_b 

       ΔV *= model.deg_params.adjustment_coef 

       #Adjust with time spent (ref time is 1 hour)
       ΔV *= Δh

       model.V_J[2,:] .-= ΔV
    end


    V_nom = interpolation(model.V_J[1,:], model.V_J[2,:], model.J_ref, true)

    if model.StartStop
      start_stop_count = get_start_stops(powers)
      model.V_J[2,:] .-= model.deg_params.start_stop_coef * V_nom * start_stop_count
    end 

    model.V_J[3,:] = model.V_J[2,:] .* (model.V_J[1,:] * fc.surface * fc.N_cell) 

    V_nom_ini = interpolation(model.V_J_ini[1,:], model.V_J_ini[2,:], model.J_ref, true)
    V_nom = interpolation(model.V_J[1,:], model.V_J[2,:], model.J_ref, true)


    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:], model.V_J[2,:])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end

    nextSoH = V_nom/V_nom_ini  

    if fc.SoC_model.couplage
      nextPowerMax = maximum(model.V_J[3,:]) * (1-fc.SoC_model.k_aux)
      nextPowerMin = compute_min_power(fc)
      fc.SoC_model.V_J = model.V_J

        if fc.SoC_model isa LinearFuelCellEfficiency 
          update_η_lin(fc, fc.SoC_model)
        end

    else
      nextPowerMax = fc.SoC_model.powerMax[h,y,s] 
      nextPowerMin = fc.SoC_model.powerMin[h,y,s] 
    end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.SoC_model.powerMax[h,y,s] 
    nextPowerMin = fc.SoC_model.powerMin[h,y,s] 
  end

  return nextSoH, nextPowerMax, nextPowerMin
end


function compute_operation_soh(fc::FuelCell, model::FunctHoursAgingFuelCell, h::Int64,  y::Int64,  s::Int64,  Δh::Int64)

  if (h%convert(Int64,floor(8760/model.update_by_year))) == 0 
    interval = (h-convert(Int64,floor(8760/model.update_by_year))+1):h
    
    powers = fc.carrier[1].power[interval,y,s]

    n_hours_active = sum(powers .> 1e-6)

    #Base degradation considered at nominal current density
    ΔV = (model.V_J[1,:] * model.coef_a) .+ model.coef_b

    ΔV *= model.deg_params.adjustment_coef

    #Adjust with time spent (ref time is 1 hour)
    ΔV *= n_hours_active * Δh
       
    model.V_J[2,:] .-= ΔV
  
  

    V_nom = interpolation(model.V_J[1,:], model.V_J[2,:], model.J_ref, true)

    if model.StartStop
      start_stop_count = get_start_stops(powers)
      model.V_J[2,:] .-= model.deg_params.start_stop_coef * V_nom * start_stop_count
    end 

    model.V_J[3,:] = model.V_J[2,:] .* (model.V_J[1,:] * fc.surface * fc.N_cell) 

    V_nom_ini = interpolation(model.V_J_ini[1,:], model.V_J_ini[2,:], model.J_ref, true)
    V_nom = interpolation(model.V_J[1,:], model.V_J[2,:], model.J_ref, true)


    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:], model.V_J[2,:])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end

    nextSoH = V_nom/V_nom_ini  

      if fc.SoC_model.couplage
        nextPowerMax = maximum(model.V_J[3,:]) * (1-fc.SoC_model.k_aux)
        nextPowerMin = compute_min_power(fc)
        fc.SoC_model.V_J = model.V_J

          if fc.SoC_model isa LinearFuelCellEfficiency 
            update_η_lin(fc, fc.SoC_model)
          end

      else
        nextPowerMax = fc.SoC_model.powerMax[h,y,s] 
        nextPowerMin = fc.SoC_model.powerMin[h,y,s] 
      end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.SoC_model.powerMax[h,y,s] 
    nextPowerMin = fc.SoC_model.powerMin[h,y,s] 
  end

  return nextSoH, nextPowerMax, nextPowerMin
end


function compute_operation_soh(fc::FuelCell, model::FixedLifetimeFuelCell, h::Int64,  y::Int64,  s::Int64,  Δh::Int64)

  if (h%convert(Int64,floor(8760/model.update_by_year))) == 0 
    duration = convert(Int64,floor(8760/model.update_by_year))
    
    frac = duration/model.nHourMax
    #Base degradation considered at nominal current density
  
    ΔV = frac * model.V_nom_ini * (1-fc.SoH_threshold)
       
    model.V_J[2,:] .-= ΔV
    
    model.V_J[3,:] = model.V_J[2,:] .* (model.V_J[1,:] * fc.surface * fc.N_cell) 

    nextSoH = fc.soh[h,y,s] - (frac * (1-fc.SoH_threshold))

    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:], model.V_J[2,:])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end


      if fc.SoC_model.couplage
        nextPowerMax = maximum(model.V_J[3,:]) * (1-fc.SoC_model.k_aux)
        nextPowerMin = compute_min_power(fc)
        fc.SoC_model.V_J = model.V_J

          if fc.SoC_model isa LinearFuelCellEfficiency 
            update_η_lin(fc, fc.SoC_model)
          end

      else
        nextPowerMax = fc.SoC_model.powerMax[h,y,s] 
        nextPowerMin = fc.SoC_model.powerMin[h,y,s] 
      end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.SoC_model.powerMax[h,y,s] 
    nextPowerMin = fc.SoC_model.powerMin[h,y,s] 
  end

  return nextSoH, nextPowerMax, nextPowerMin
end


function initialize_investments!(s::Int64, fc::FuelCell, decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})

  fc.surface = decision.surface
  fc.N_cell = decision.N_cell
  fc.soh[1,1,s] = fc.soh_ini


  fc.SoC_model.V_J[1,:] = fc.V_J_ini[1,:] 
  fc.SoC_model.V_J[2,:] = fc.V_J_ini[2,:]
  fc.SoC_model.V_J[3,:] = fc.V_J_ini[2,:] .* fc.V_J_ini[1,:] * fc.surface * fc.N_cell
  fc.SoC_model.powerMax[1,1,s] = maximum(fc.SoC_model.V_J[3,:]) * (1-fc.SoC_model.k_aux)
  fc.SoC_model.powerMin[1,1,s] = compute_min_power(fc)

  if fc.SoC_model isa LinearFuelCellEfficiency 
    update_η_lin(fc, fc.SoC_model)
  end

  #Initialization of V(J)
    fc.SoH_model.V_J_ini[1,:] = fc.V_J_ini[1,:] 
    fc.SoH_model.V_J_ini[2,:] = fc.V_J_ini[2,:]
    fc.SoH_model.V_J_ini[3,:] = fc.V_J_ini[2,:] .* fc.V_J_ini[1,:] * fc.surface * fc.N_cell
    fc.SoH_model.V_J = copy(fc.SoH_model.V_J_ini)

  if fc.SoH_model isa FunctHoursAgingFuelCell

    fc.SoH_model.coef_b = fc.SoH_model.deg_params.b
    fc.SoH_model.coef_a = get_slope_deg(fc.SoH_model.J_base, fc.SoH_model.deg_params.power_slope, fc.SoH_model.deg_params.a_slope, fc.SoH_model.deg_params.b_slope, fc.SoH_model.deg_params.c_slope)
  
  elseif fc.SoH_model isa FixedLifetimeFuelCell

    fc.SoH_model.V_nom_ini = interpolation(fc.V_J_ini[1,:], fc.V_J_ini[2,:], fc.SoH_model.J_ref , true)

  end
  
end




  ### Investment dynamic
  function compute_investment_dynamics!(y::Int64, s::Int64, fc::FuelCell,  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})    
    fc.SoC_model.powerMax[1,y+1,s], fc.SoC_model.powerMin[1,y+1,s], fc.soh[1,y+1,s] = compute_investment_dynamics(fc, (powerMax = fc.SoC_model.powerMax[end,y,s], powerMin = fc.SoC_model.powerMin[end,y,s], soh = fc.soh[end,y,s]), decision)
  end

  
  function compute_investment_dynamics(fc::FuelCell, state::NamedTuple{(:powerMax, :powerMin, :soh), Tuple{Float64, Float64, Float64}},  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
    if decision.N_cell > 1e-2 

        V_J = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P

        fc.surface = decision.surface
        fc.N_cell = decision.N_cell
        

        for (i,a) in enumerate([fc.V_J_ini[1,:], fc.V_J_ini[2,:], fc.V_J_ini[2,:] .* fc.V_J_ini[1,:] * fc.surface * fc.N_cell])
            V_J[i,:] = a 
        end

        fc.SoH_model.J_ref = 0.62

        soh_next = fc.soh_ini

        fc.SoH_model.V_J = V_J
        fc.SoC_model.V_J = V_J

        powerMax_next = maximum(V_J[3,:]) * (1-fc.SoC_model.k_aux)

        powerMin_next = compute_min_power(fc)

        if fc.SoC_model isa LinearFuelCellEfficiency 
          update_η_lin(fc, fc.SoC_model)
        end

    else
        powerMax_next = state.powerMax
        powerMin_next = state.powerMin

        soh_next = state.soh
    end

    return powerMax_next, powerMin_next, soh_next
end

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



function get_η_E(P_net::Float64, fc::FuelCell)
  P_tot = floor(P_net / (1 - fc.SoC_model.k_aux); digits=6)
  
  #Find the corresponding current from an interpolation from P(I) curve 
  j = interpolation(fc.SoC_model.V_J[3,:], fc.SoC_model.V_J[1,:], P_tot, true )
  i = j * fc.surface

  return P_net / (fc.SoC_model.K * i * fc.N_cell)

end




#compute the power that correpond to the maximum allowed tension
function compute_min_power(fc::FuelCell)
  if fc.SoC_model isa FixedFuelCellEfficiency
    P_min = fc.SoC_model.α_p * maximum(fc.SoC_model.V_J[3,:])
  else
    P_min_tot = interpolation(fc.SoC_model.V_J[2,:], fc.SoC_model.V_J[3,:], fc.SoC_model.V_max, false )
    P_min = P_min_tot / (1 + fc.SoC_model.k_aux)
  end
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


function update_η_lin(fc::FuelCell, model::LinearFuelCellEfficiency)
  P_max = maximum(model.V_J[3,:]) * (1-model.k_aux)
  P_min = compute_min_power(fc)
  
  η_P_min = get_η_E(P_min, fc)
  η_P_max = get_η_E(P_max, fc)
  
  a_η = (η_P_max - η_P_min) / (P_max - P_min)
  b_η = η_P_min - a_η * P_min


  fc.SoC_model.a_η = a_η
  fc.SoC_model.b_η = b_η

end
