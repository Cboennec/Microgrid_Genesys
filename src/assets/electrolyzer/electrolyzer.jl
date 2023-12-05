abstract type AbstractElectrolyzer <: AbstractConverter  end


abstract type AbstractElectrolyzerEffModel end

abstract type AbstractElectrolyzerAgingModel end



mutable struct FixedElectrolyzerEfficiency <: AbstractElectrolyzerEffModel
 
  α_p::Float64 #Minimum power defined as a share of the maximum Power
  η_E_H2::Float64 #The efficiency from Electricity to DiHydrogen 
  η_E_H::Float64 #The efficiency from Electricity to Heat
  k_aux::Float64
  powerMax_ini::Float64
  couplage::Bool 

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell

  V_J::AbstractArray{Float64,2}


  FixedElectrolyzerEfficiency(;α_p = 0.08,
    η_E_H2 = 0.7,
    η_E_H = 0.,
    k_aux = 0.15,
    powerMax_ini =.00001,
    couplage = false
    ) = new(α_p, η_E_H2, η_E_H, k_aux, powerMax_ini, couplage)
end

mutable struct PolarizationElectrolyzerEfficiency <: AbstractElectrolyzerEffModel
  k_aux::Float64 # Share of the power used by the auxiliaries
  couplage::Bool
  K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
  powerMax_ini::Float64
  J_min::Float64

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the Electrolyzer

  V_J::AbstractArray{Float64,2}



  PolarizationElectrolyzerEfficiency(; k_aux = 0.15,
  couplage = true,
  K = (33.33 *  2.016 * 3600)  / (2*96485.3321),  #LHV * M_H2 * λ * 3600/(2*F)
  powerMax_ini = .00001,
  J_min = 0.1
  ) = new(k_aux, couplage, K, powerMax_ini, J_min)
end


mutable struct LinearElectrolyzerEfficiency <: AbstractElectrolyzerEffModel
  k_aux::Float64 # Share of the power used by the auxiliaries
  couplage::Bool
  K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
  powerMax_ini::Float64
  J_min::Float64

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the Electrolyzer

  a_η::Float64 # the slope for the fucntion η(P)
  b_η::Float64 # the ordinate at the origin for the function η(P)

  V_J::AbstractArray{Float64,2}


  LinearElectrolyzerEfficiency(; k_aux = 0.15,
  couplage = true,
  K = (33.33 *  2.016 * 3600)  / (2*96485.3321),  #PCI * M_H2 * 3600/(2*F)
  powerMax_ini = .00001,
  J_min = 0.1
  ) = new(k_aux, couplage, K, powerMax_ini, J_min)
end



#M/A en option
mutable struct FunctHoursAgingElectrolyzer <: AbstractElectrolyzerAgingModel
  plot::Bool 
  StartStop::Bool
  deg_per_hour::Float64 
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current density
  J_base::Float64 # The current density used for degradation

  V_J_ini::AbstractArray{Float64,2}
  V_J::AbstractArray{Float64,2}

  coef_a::Float64 #The slope of voltage degradation for each functioning hour
  coef_b::Float64 #The ordinate at origin of voltage degradation for each functioning hour

  FunctHoursAgingElectrolyzer(;plot = false,
    StartStop = true,
    deg_per_hour = 1e-5,
    update_by_year = 12,
    J_ref = 0.62,
    J_base = 0.1
  ) = new(plot, StartStop, deg_per_hour, update_by_year, J_ref, J_base)
end



#M/A en option
mutable struct FixedLifetimeElectrolyzer <: AbstractElectrolyzerAgingModel
  plot::Bool 
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current density
  nHourMax::Int64

  V_J_ini::AbstractArray{Float64,2}
  V_J::AbstractArray{Float64,2}

  V_nom_ini::Float64

  FixedLifetimeElectrolyzer(;plot = false,
    update_by_year = 12,
    J_ref = 0.62,
    nHourMax = 20*8760
  ) = new(plot, update_by_year, J_ref, nHourMax)
end





mutable struct Electrolyzer <: AbstractElectrolyzer

  EffModel::AbstractElectrolyzerEffModel
	SoH_model::AbstractElectrolyzerAgingModel
	
	bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
	SoH_threshold::Float64 # SoH level to replace battery
	couplage::Bool  #a boolean tuple to tell wether or not the soh should influence the other parameters.
  V_J_ini::AbstractArray{Float64,2}

  min_part_load::Float64 #Minimum share of the nominal power to be activated

	# Initial conditions
	soh_ini::Float64 # first state of health for the begining of simulation

  N_cell::Int64 #The number of assembled cells
  surface::Float64 #The number of assembled cells



  η::AbstractArray{Float64,3}
  carrier::Vector{EnergyCarrier}
  soh::AbstractArray{Float64,3}

  # Eco
  cost::AbstractArray{Float64,2}

	Electrolyzer(;EffModel = PolarizationElectrolyzerEfficiency(),
    SoH_model = PowerAgingElectrolyzer(),
    bounds = (lb = 0., ub = 50.),
    SoH_threshold = 0.8,
    couplage = true,
    V_J_ini = nothing,
    min_part_load = 0.05,
    soh_ini = 1. 
  ) = new(EffModel, SoH_model, bounds, SoH_threshold, couplage, V_J_ini, min_part_load, soh_ini)

end






### Preallocation
function preallocate!(elyz::Electrolyzer, nh::Int64, ny::Int64, ns::Int64)
  elyz.EffModel.powerMax = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  elyz.EffModel.powerMax[1,1,:] .= elyz.EffModel.powerMax_ini
  elyz.η = convert(SharedArray,zeros(nh+1, ny+1, ns))

  elyz.carrier = [Electricity(), Heat(), Hydrogen()]
  elyz.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
  elyz.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
  elyz.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
  elyz.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; elyz.soh[1,1,:] .= elyz.soh_ini
  elyz.cost = convert(SharedArray,zeros(ny, ns))

  elyz.SoH_model.V_J_ini = zeros(3, length(elyz.V_J_ini[1,:])) #J, V, P
  elyz.SoH_model.V_J = zeros(3, length(elyz.V_J_ini[1,:])) #J, V, P
  elyz.EffModel.V_J = zeros(3, length(elyz.V_J_ini[1,:])) #J, V, P

  return elyz
end


### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, elyz::Electrolyzer, decision::Float64, Δh::Int64)

  elyz.carrier[1].power[h,y,s], elyz.carrier[2].power[h,y,s], elyz.carrier[3].power[h,y,s] = compute_operation_efficiency(elyz, elyz.EffModel, h ,y ,s , decision)

  elyz.soh[h+1,y,s], elyz.EffModel.powerMax[h+1,y,s] = compute_operation_soh(elyz, elyz.SoH_model, h ,y ,s, Δh)

end


### Operation dynamic
function compute_operation_dynamics(elyz::Electrolyzer, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)

  return compute_operation_efficiency(elyz, elyz.EffModel, h ,y ,s , decision)

end



function compute_operation_efficiency(elyz::Electrolyzer, model::FixedElectrolyzerEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64)
	
  #Apply minimum power
  model.powerMax[h,y,s] * elyz.min_part_load <= -decision ? power_E = max(decision, -model.powerMax[h,y,s]) : power_E = 0. 
       
   if power_E < 0
      elec_power, heat_power, hydrogen_power = (power_E), - power_E * model.η_E_H,  - power_E * model.η_E_H2
      elyz.η[h,y,s] = hydrogen_power / -elec_power
   else 
       
     elec_power, heat_power, hydrogen_power = 0., 0., 0.
   end
 
   return elec_power, heat_power, hydrogen_power 
 end




function compute_operation_efficiency(elyz::Electrolyzer, model::PolarizationElectrolyzerEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64)

  #Apply minimum power
  model.powerMax[h,y,s] * elyz.min_part_load <= -decision ? power_E = max(decision, -model.powerMax[h,y,s]) : power_E = 0. 
  if power_E < 0
    #Compute the remaining power after feeding the auxiliaries 

    P_elyz = ceil(power_E / (1 + model.k_aux); digits=6)

    j = interpolation(model.V_J[3,:], model.V_J[1, :], -P_elyz, true)
    
    i = j * elyz.surface

    #Power_out/power_in (per cell)
    η_E_H2 = i * model.K / (-power_E/ elyz.N_cell)     

    elyz.η[h,y,s] = η_E_H2

    η_E_H = 0.
   
    elec_power, heat_power, hydrogen_power = (power_E), - power_E * η_E_H,  - power_E * η_E_H2

  else 
    elec_power, heat_power, hydrogen_power = 0., 0., 0.
  end
 
  return elec_power, heat_power, hydrogen_power 
end



function compute_operation_efficiency(elyz::Electrolyzer, model::LinearElectrolyzerEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64)

  #Apply minimum power
  model.powerMax[h,y,s] * elyz.min_part_load <= -decision ? power_E = max(decision, -model.powerMax[h,y,s]) : power_E = 0. 
  if power_E < 0
    #Compute the remaining power after feeding the auxiliaries 

    η_E_H2 = model.a_η * -power_E + model.b_η

    elyz.η[h,y,s] = η_E_H2

    η_E_H = 0
   
    elec_power, heat_power, hydrogen_power = (power_E), - power_E * η_E_H,  - power_E * η_E_H2

  else 
    elec_power, heat_power, hydrogen_power = 0., 0., 0.
  end
 
  return elec_power, heat_power, hydrogen_power 
end



function compute_operation_soh(elyz::Electrolyzer, model::FunctHoursAgingElectrolyzer, h::Int64,  y::Int64,  s::Int64,  Δh::Int64)

  if (h%convert(Int64,floor(8760/model.update_by_year))) == 0 
    interval = (h-convert(Int64,floor(8760/model.update_by_year))+1):h
    
    powers = elyz.carrier[1].power[interval,y,s]

    hours_funct = sum(powers .< -1e-6)

    ΔV = hours_funct * model.deg_per_hour * Δh
       
    model.V_J[2,:] .-= ΔV
  
    V_nom = interpolation(model.V_J[1,:], model.V_J[2,:], model.J_ref, true)

    model.V_J[3,:] = model.V_J[2,:] .* (model.V_J[1,:] * elyz.surface * elyz.N_cell) 

    V_nom_ini = interpolation(model.V_J_ini[1,:], model.V_J_ini[2,:], model.J_ref, true)

    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:], model.V_J[2,:])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end

    nextSoH = V_nom/V_nom_ini  

      if elyz.EffModel.couplage
        nextPowerMax = maximum(model.V_J[3,:]) * (1-elyz.EffModel.k_aux)
        elyz.EffModel.V_J = model.V_J

          if elyz.EffModel isa LinearFuelCellEfficiency 
            update_η_lin(elyz, elyz.EffModel)
          end

      else
        nextPowerMax = elyz.EffModel.powerMax[h,y,s] 
      end
  else 
    nextSoH = elyz.soh[h,y,s] 
    nextPowerMax = elyz.EffModel.powerMax[h,y,s] 
  end

  return nextSoH, nextPowerMax
end



function compute_operation_soh(elyz::Electrolyzer, model::FixedLifetimeElectrolyzer, h::Int64,  y::Int64,  s::Int64,  Δh::Int64)

  if (h%convert(Int64,floor(8760/model.update_by_year))) == 0 
    duration = convert(Int64,floor(8760/model.update_by_year))
    
    frac = duration/model.nHourMax
    #Base degradation considered at nominal current density
  
    ΔV = frac * model.V_nom_ini * (1-elyz.SoH_threshold)
       
    model.V_J[2,:] .-= ΔV
    
    model.V_J[3,:] = model.V_J[2,:] .* (model.V_J[1,:] * elyz.surface * elyz.N_cell) 

    nextSoH = elyz.soh[h,y,s] - (frac * (1-elyz.SoH_threshold))

    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:], model.V_J[2,:])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end


      if elyz.EffModel.couplage
        nextPowerMax = maximum(model.V_J[3,:]) * (1-elyz.EffModel.k_aux)
        nextPowerMin = compute_min_power(elyz)
        elyz.EffModel.V_J = model.V_J

          if elyz.EffModel isa LinearElectrolyzerEfficiency 
            update_η_lin(elyz, elyz.EffModel)
          end

      else
        nextPowerMax = elyz.EffModel.powerMax[h,y,s] 
      end
  else 
    nextSoH = elyz.soh[h,y,s] 
    nextPowerMax = elyz.EffModel.powerMax[h,y,s] 
  end

  return nextSoH, nextPowerMax
end

function initialize_investments!(s::Int64, elyz::Electrolyzer, decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})

  elyz.surface = decision.surface
  elyz.N_cell = decision.N_cell
  elyz.soh[1,1,s] = elyz.soh_ini


  elyz.EffModel.V_J[1,:] = elyz.V_J_ini[1,:] 
  elyz.EffModel.V_J[2,:] = elyz.V_J_ini[2,:]
  elyz.EffModel.V_J[3,:] = elyz.V_J_ini[2,:] .* elyz.V_J_ini[1,:] * elyz.surface * elyz.N_cell
  elyz.EffModel.powerMax[1,1,s] = maximum(elyz.EffModel.V_J[3,:]) * (1-elyz.EffModel.k_aux)

  if elyz.EffModel isa LinearElectrolyzerEfficiency 
    update_η_lin(elyz, elyz.EffModel)
  end

  #Initialization of V(J)
    elyz.SoH_model.V_J_ini[1,:] = elyz.V_J_ini[1,:] 
    elyz.SoH_model.V_J_ini[2,:] = elyz.V_J_ini[2,:]
    elyz.SoH_model.V_J_ini[3,:] = elyz.V_J_ini[2,:] .* elyz.V_J_ini[1,:] * elyz.surface * elyz.N_cell
    elyz.SoH_model.V_J = copy(elyz.SoH_model.V_J_ini)

  if elyz.SoH_model isa FixedLifetimeElectrolyzer

    elyz.SoH_model.V_nom_ini = interpolation(elyz.V_J_ini[1,:], elyz.V_J_ini[2,:], elyz.SoH_model.J_ref , true)

  end


  
end

### Investment dynamic
function compute_investment_dynamics!(y::Int64, s::Int64, elyz::Electrolyzer,  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})    
  elyz.EffModel.powerMax[1,y+1,s], elyz.soh[1,y+1,s] = compute_investment_dynamics(elyz, (powerMax = elyz.EffModel.powerMax[end,y,s], soh = elyz.soh[end,y,s]), decision)
end


function compute_investment_dynamics(elyz::Electrolyzer, state::NamedTuple{(:powerMax, :soh), Tuple{Float64, Float64}},  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})
  if decision.N_cell > 1e-2 

      V_J = zeros(3, length(elyz.V_J_ini[1,:])) #J, V, P

      elyz.surface = decision.surface
      elyz.N_cell = decision.N_cell
      

      for (i,a) in enumerate([elyz.V_J_ini[1,:], elyz.V_J_ini[2,:], elyz.V_J_ini[2,:] .* elyz.V_J_ini[1,:] * elyz.surface * elyz.N_cell])
          V_J[i,:] = a 
      end

      elyz.SoH_model.J_ref = 1

      soh_next = elyz.soh_ini

      elyz.SoH_model.V_J = V_J
      elyz.EffModel.V_J = V_J

      powerMax_next = maximum(V_J[3,:]) * (1-elyz.EffModel.k_aux)

      if elyz.EffModel isa LinearFuelCellEfficiency 
        update_η_lin(elyz, elyz.EffModel)
      end

  else
      powerMax_next = state.powerMax

      soh_next = state.soh
  end

  return powerMax_next, soh_next
end



#compute the power that correpond to the maximum allowed tension
function compute_min_power(elyz::AbstractElectrolyzer)
  if elyz.EffModel isa FixedElectrolyzerEfficiency
    P_min = elyz.EffModel.α_p * maximum(elyz.EffModel.V_J[3,:])
  else
    P_min = interpolation(elyz.EffModel.V_J[1,:], elyz.EffModel.V_J[3,:], elyz.EffModel.J_min, true )
  end
    P_min_tot = P_min * (1 + elyz.EffModel.k_aux)
  
  return P_min_tot
end


function get_η_E(P_brut::Float64, elyz::AbstractElectrolyzer)
  P_net = ceil(P_brut / (1 + elyz.EffModel.k_aux); digits=6)
  
  #Find the corresponding current from an interpolation from P(I) curve 
  j = interpolation(elyz.EffModel.V_J[3,:], elyz.EffModel.V_J[1,:], P_net, true)
  i = j * elyz.surface * elyz.N_cell

  return elyz.EffModel.K * i / (P_brut)

end


function update_η_lin(elyz::Electrolyzer, model::LinearElectrolyzerEfficiency)
  P_max = maximum(model.V_J[3,:]) * (1-model.k_aux)
  P_min = compute_min_power(elyz)
  
  η_P_min = get_η_E(P_min, elyz)
  η_P_max = get_η_E(P_max, elyz)
  
  a_η = (η_P_max - η_P_min) / (P_max - P_min)
  b_η = η_P_min - a_η * P_min


  elyz.EffModel.a_η = a_η
  elyz.EffModel.b_η = b_η

end



function toStringShort(elyz::Electrolyzer)

	if elyz.EffModel isa LinearElectrolyzerEfficiency
		efficiency = "x"
	elseif elyz.EffModel isa PolarizationElectrolyzerEfficiency
		efficiency = "V(J)"
  elseif elyz.EffModel isa FixedElectrolyzerEfficiency
		efficiency = "fix."
	end

	if elyz.SoH_model isa FixedLifetimeElectrolyzer
		aging = "FL"
	elseif elyz.SoH_model isa FunctHoursAgingElectrolyzer
		aging = "FH"
	end

	return string("Elyz :", efficiency, ", ", aging)
end