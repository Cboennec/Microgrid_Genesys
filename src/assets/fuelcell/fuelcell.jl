abstract type AbstractFuelCell <: AbstractConverter  end

abstract type AbstractFuelCellEffModel end

abstract type AbstractFuelCellAgingModel end




"""
FixedFuelCellEfficiency

A mutable struct representing a fixed efficiency fuel cell model.

# Parameters:
- `α_p::Float64`: Minimum power defined as a share of the maximum power (default: 0.08)
- `η_H2_E::Float64`: Efficiency from DiHydrogen to Electricity (default: 0.4)
- `η_H2_H::Float64`: Efficiency from DiHydrogen to Heat (default: 0.4)
- `k_aux::Float64`: Auxiliary power coefficient (default: 0.15)
- `powerMax_ini::Float64`: Initial maximum power (default: 0.00001)
- `couplage::Bool`: Boolean indicating if there is coupling between parameters (default: false)

# Arrays:
- `powerMax::AbstractArray{Float64,3}`: The maximum power that can be demanded from the fuel cell
- `powerMin::AbstractArray{Float64,3}`: The minimum power that can be demanded from the fuel cell
- `V_J::AbstractArray{Float64,2}`: polarization curve (voltage as a function of current density)

## Example 
```julia
FixedFuelCellEfficiency()
```
"""
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


"""
PolarizationFuelCellEfficiency

A mutable struct representing a fuel cell efficiency model based on polarization curve.

# Parameters:
- `k_aux::Float64`: Share of the power used by auxiliaries (default: 0.15)
- `couplage::Bool`: Boolean indicating if there is coupling between parameters (default: true)
- `K::Float64`: Constant value calculated as Latent Heat Value * molar mass * stoichiometric coefficient / (2 * Faraday constant) (default: (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321))
- `powerMax_ini::Float64`: Initial maximum power (default: 0.00001)
- `V_max::Float64`: Maximum voltage (default: 0.8 V)

# Arrays:
- `powerMax::AbstractArray{Float64,3}`: The maximum power that can be demanded from the fuel cell
- `powerMin::AbstractArray{Float64,3}`: The minimum power that can be demanded from the fuel cell
- `V_J::AbstractArray{Float64,2}`: Polarization curve (voltage as a function of current density)

## Example 
```julia
PolarizationFuelCellEfficiency()
```
"""
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

"""
LinearFuelCellEfficiency

A mutable struct representing a linear fuel cell efficiency model.

# Parameters:
- `k_aux::Float64`: Share of the power used by auxiliaries (default: 0.15)
- `couplage::Bool`: Boolean indicating if there is coupling between parameters (default: true)
- `K::Float64`: Constant value calculated as Latent Heat Value * molar mass * stoichiometric coefficient / (2 * Faraday constant) (default: (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321))
- `powerMax_ini::Float64`: Initial maximum power (default: 0.00001)
- `V_max::Float64`: Maximum voltage (default: 0.8)
- `a_η::Float64`: Slope for the function η(P) 
- `b_η::Float64`: Ordinate at the origin for the function η(P) 

# Arrays:
- `powerMax::AbstractArray{Float64,3}`: The maximum power that can be demanded from the fuel cell
- `powerMin::AbstractArray{Float64,3}`: The minimum power that can be demanded from the fuel cell
- `V_J::AbstractArray{Float64,2}`: Polarization curve (voltage as a function of current density)

## Example 
```julia
LinearFuelCellEfficiency()
```
"""
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


"""
deg_params

A mutable struct containing parameters for aging and the modification of the polarization curve. Aging is computed based on the evolution of the polarization curve, and this struct provides the parameters for describing it.

# Parameters:
- `a_slope::Float64`, `b_slope::Float64`, `c_slope::Float64`: Coefficients for computing the degradation slope (of the affine degradation) as: a * j^power + b * j + c
- `power_slope::Float64`: The power in the slope equation
- `b::Float64`: Ordinate at origin
- `adjustment_coef::Float64`: Coefficient for achieving a life expectancy of X hours at the reference current density
- `start_stop_coef::Float64`: Degradation due to starts and stops

## Example 
```julia
deg_params()
```
"""
mutable struct deg_params
  #Structure containing the parameters for the aging and the modification of th epolarization curve. Aging gonna be computed based on the evolution of the polarization curve and this struct gives the paramters for describing it.
  #At each hour the degradation is an afine function defining the number of μV/h to withdraw from the polarization curve 
  a_slope::Float64 # a, b and c coef for computing the degradation slope as : a j^power + b j + c 
  b_slope::Float64
  c_slope::Float64
  power_slope::Float64 # The power in the slope equation
  b::Float64 #Ordinate at origin
  adjustment_coef::Float64 # Coef for having a life expectency of X hours at ref current desnity
  start_stop_coef::Float64 # Degradation due to start and stops
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

  EffModel::AbstractFuelCellEffModel
	SoH_model::AbstractFuelCellAgingModel
	
	bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
	SoH_threshold::Float64 # SoH level to replace battery
	couplage::Bool  #a boolean tuple to tell wether or not the soh should influence the other parameters.
  V_J_ini::AbstractArray{Float64,2}

	# Initial conditions
	soh_ini::Float64 # first state of health for the begining of simulation

  N_cell::Int64 #The number of assembled cells
  surface::Float64 #The number of assembled cells



  η::AbstractArray{Float64,3}
  carrier::Vector{EnergyCarrier}
  soh::AbstractArray{Float64,3}

  # Eco
  cost::AbstractArray{Float64,2}

	FuelCell(;EffModel = PolarizationFuelCellEfficiency(),
    SoH_model = PowerAgingFuelCell(),
    bounds = (lb = 0., ub = 50.),
    SoH_threshold = 0.9,
    couplage = true,
    V_J_ini = nothing,
    soh_ini = 1. 
  ) = new(EffModel, SoH_model, bounds, SoH_threshold, couplage, V_J_ini, soh_ini)

end



  
  ### Preallocation
  function preallocate!(fc::FuelCell, nh::Int64, ny::Int64, ns::Int64)

    fc.EffModel.powerMax = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  fc.EffModel.powerMax[1,1,:] .= fc.EffModel.powerMax_ini
    fc.EffModel.powerMin = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  fc.EffModel.powerMin[1,1,:] .= fc.EffModel.powerMax_ini
    fc.η = convert(SharedArray,zeros(nh+1, ny+1, ns))
    fc.carrier = [Electricity(), Heat(), Hydrogen()]
    fc.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; fc.soh[1,1,:] .= fc.soh_ini
    fc.cost = convert(SharedArray,zeros(ny, ns))

    fc.SoH_model.V_J_ini = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P
    fc.SoH_model.V_J = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P
    fc.EffModel.V_J = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P

    return fc
end
  

### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, fc::FuelCell, decision::Float64, Δh::Int64)

  fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] = compute_operation_efficiency(fc, fc.EffModel, h ,y ,s , decision, Δh)

  fc.soh[h+1,y,s], fc.EffModel.powerMax[h+1,y,s], fc.EffModel.powerMin[h+1,y,s] = compute_operation_soh(fc, fc.SoH_model, h ,y ,s, Δh)

end


### Operation dynamic
function compute_operation_dynamics(fc::FuelCell, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)

  return compute_operation_efficiency(fc, fc.EffModel, h ,y ,s , decision, Δh)

end


function compute_operation_efficiency(fc::FuelCell, model::PolarizationFuelCellEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
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



function compute_operation_efficiency(fc::FuelCell, model::FixedFuelCellEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
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


 
function compute_operation_efficiency(fc::FuelCell, model::LinearFuelCellEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
  #Apply minimum power
   model.powerMin[h,y,s] <= decision ? power_E = min(decision, model.powerMax[h,y,s]) : power_E = 0. 
       
   if power_E > 0
      #Compute the power needed to feed the auxiliaries and deliver the right power
      P_tot = floor(power_E / (1 - model.k_aux); digits=6)
 
      η_E = model.a_η * P_tot + model.b_η 
      
      if η_E >= 0.45 && y==9 && h < 1000
        println("y,h = ", y, ", ", h, "   a_η = ", model.a_η, " , P_tot = ", P_tot, ", b_η = ",  model.b_η )
      end


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

    if fc.EffModel.couplage
      nextPowerMax = maximum(model.V_J[3,:]) * (1-fc.EffModel.k_aux)
      nextPowerMin = compute_min_power(fc)
      fc.EffModel.V_J = model.V_J

        if fc.EffModel isa LinearFuelCellEfficiency 
          update_η_lin(fc, fc.EffModel)
        end

    else
      nextPowerMax = fc.EffModel.powerMax[h,y,s] 
      nextPowerMin = fc.EffModel.powerMin[h,y,s] 
    end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.EffModel.powerMax[h,y,s] 
    nextPowerMin = fc.EffModel.powerMin[h,y,s] 
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

      if fc.EffModel.couplage
        nextPowerMax = maximum(model.V_J[3,:]) * (1-fc.EffModel.k_aux)
        nextPowerMin = compute_min_power(fc)
        fc.EffModel.V_J = model.V_J

          if fc.EffModel isa LinearFuelCellEfficiency 
            update_η_lin(fc, fc.EffModel)
          end

      else
        nextPowerMax = fc.EffModel.powerMax[h,y,s] 
        nextPowerMin = fc.EffModel.powerMin[h,y,s] 
      end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.EffModel.powerMax[h,y,s] 
    nextPowerMin = fc.EffModel.powerMin[h,y,s] 
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


      if fc.EffModel.couplage
        nextPowerMax = maximum(model.V_J[3,:]) * (1-fc.EffModel.k_aux)
        nextPowerMin = compute_min_power(fc)
        fc.EffModel.V_J = model.V_J

          if fc.EffModel isa LinearFuelCellEfficiency 
            update_η_lin(fc, fc.EffModel)
          end

      else
        nextPowerMax = fc.EffModel.powerMax[h,y,s] 
        nextPowerMin = fc.EffModel.powerMin[h,y,s] 
      end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.EffModel.powerMax[h,y,s] 
    nextPowerMin = fc.EffModel.powerMin[h,y,s] 
  end

  return nextSoH, nextPowerMax, nextPowerMin
end


function initialize_investments!(s::Int64, fc::FuelCell, decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})

  fc.surface = decision.surface
  fc.N_cell = decision.N_cell
  fc.soh[1,1,s] = fc.soh_ini


  fc.EffModel.V_J[1,:] = fc.V_J_ini[1,:] 
  fc.EffModel.V_J[2,:] = fc.V_J_ini[2,:]
  fc.EffModel.V_J[3,:] = fc.V_J_ini[2,:] .* fc.V_J_ini[1,:] * fc.surface * fc.N_cell
  fc.EffModel.powerMax[1,1,s] = maximum(fc.EffModel.V_J[3,:]) * (1-fc.EffModel.k_aux)
  fc.EffModel.powerMin[1,1,s] = compute_min_power(fc)

  if fc.EffModel isa LinearFuelCellEfficiency 
    update_η_lin(fc, fc.EffModel)
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
    fc.EffModel.powerMax[1,y+1,s], fc.EffModel.powerMin[1,y+1,s], fc.soh[1,y+1,s] = compute_investment_dynamics(fc, (powerMax = fc.EffModel.powerMax[end,y,s], powerMin = fc.EffModel.powerMin[end,y,s], soh = fc.soh[end,y,s]), decision)
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
        fc.EffModel.V_J = V_J

        powerMax_next = maximum(V_J[3,:]) * (1-fc.EffModel.k_aux)

        powerMin_next = compute_min_power(fc)

        if fc.EffModel isa LinearFuelCellEfficiency 
          update_η_lin(fc, fc.EffModel)
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
  P_tot = floor(P_net / (1 - fc.EffModel.k_aux); digits=6)
  
  #Find the corresponding current from an interpolation from P(I) curve 
  j = interpolation(fc.EffModel.V_J[3,:], fc.EffModel.V_J[1,:], P_tot, true )
  i = j * fc.surface

  return P_net / (fc.EffModel.K * i * fc.N_cell)

end




#compute the power that correpond to the maximum allowed tension
function compute_min_power(fc::FuelCell)
  if fc.EffModel isa FixedFuelCellEfficiency
    P_min = fc.EffModel.α_p * maximum(fc.EffModel.V_J[3,:])
  else
    P_min_tot = interpolation(fc.EffModel.V_J[2,:], fc.EffModel.V_J[3,:], fc.EffModel.V_max, false )
    P_min = P_min_tot / (1 + fc.EffModel.k_aux)
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


  fc.EffModel.a_η = a_η
  fc.EffModel.b_η = b_η

end


function toStringShort(fc::FuelCell)

	if fc.EffModel isa LinearFuelCellEfficiency
		efficiency = "x"
	elseif fc.EffModel isa PolarizationFuelCellEfficiency
		efficiency = "V(J)"
  elseif fc.EffModel isa FixedFuelCellEfficiency
		efficiency = "fix."
	end

	if fc.SoH_model isa PowerAgingFuelCell
		aging = "P"
	elseif fc.SoH_model isa FixedLifetimeFuelCell
		aging = "FL"
	elseif fc.SoH_model isa FunctHoursAgingFuelCell
		aging = "FH"
	end

	return string("FC :", efficiency, ", ", aging)
end


function create_deg_params(datas::Vector{DataFrames.DataFrame}, Js::Vector{Float64}, V_J::Matrix{Float64}, J_ref::Float64, objective_lifetime::Float64; power = 1/2)

  #Maximum deg profil in Alexandra Pessot thesis
  P_max = datas[3]

  #Get affine coef from input data
  as, b = fit_all_curves(datas, Js)

  #get a,b,c coef to be able to write the degradation as ax^power + bx + c
  a_slope, b_slope, c_slope = fit_dot(Js, as, power)

  #initial voltage at ref current density
  V_ini_ref =  interpolation(V_J[1,:],  V_J[2,:], J_ref, true)

  ΔV_tot = V_ini_ref * 0.1

  #Voltage ref loss
  V_deg_ref = interpolation(P_max.J,  P_max.V, J_ref, true)

  #Adujst the lifetime to fit a target (The FC will be able to be used at its ref current density for the target amount of hour)
  current_lifetime = ΔV_tot / (V_deg_ref * 1e-6)

  adaptation_coefficient = current_lifetime * 1e-6/objective_lifetime

  start_stop_coef = 0.0000196 #0.00196% as stated in Pucheng Pei, Qianfei Chang, Tian Tang,
 # A quick evaluating method for automotive fuel cell lifetime (https://doi.org/10.1016/j.ijhydene.2008.04.048)
  
  return deg_params(a_slope, b_slope, c_slope, power, b, adaptation_coefficient, start_stop_coef)

end



   
  


function fit_all_curves(data, Js)

  n_data = length(data)
  n_data_point = [length(data[i].J) for i in 1:n_data]


  m2 = Model(Ipopt.Optimizer)
  #set_optimizer_attribute(m2, "TimeLimit", 100)
  
  @variable(m2, a[1:n_data] >= 0)
  @variable(m2, b >= 0)

  @variable(m2, errors[1:n_data])
  @constraint(m2, [d in 1:n_data], errors[d] >= sum( (a[d]*data[d].J[i]+b - data[d].V[i])^2 for i in 1:n_data_point[d]))

  #Minimize the squared error
  @objective(m2, Min, sum(errors[d] for d in 1:n_data))


  optimize!(m2)
  plt = PyPlot.subplot()
  for d in 1:n_data
      PyPlot.scatter(data[d].J, data[d].V, label = string(Js[d] , " A/cm² : data"))
      PyPlot.plot(data[d].J, data[d].J .* value.(m2[:a])[d] .+ value(m2[:b]), label = string(Js[d] , " A/cm² : model"))
  end
  plt.set_xlabel("Current density (A/cm²)", fontsize=20)
  plt.set_ylabel("μV/h", fontsize=20)
  PyPlot.legend(fontsize = 15)

  return value.(m2[:a]), value(m2[:b])
end


function fit_dot(Js, as, power)

  n_data_point = length(Js)

  m2 = Model(Ipopt.Optimizer)
  #set_optimizer_attribute(m2, "TimeLimit", 100)
  
  @variable(m2, a)
  @variable(m2, b)
  @variable(m2, c)


  @variable(m2, errors[1:n_data_point])
  @constraint(m2, [i in 1:n_data_point], errors[i] == (a*Js[i]^(power)+(b * Js[i]) + c) - as[i])

  #Minimize the squared error
  @objective(m2, Min, sum((errors[i]^2) for i in 1:n_data_point))


  optimize!(m2)

  p = PyPlot.subplot()

 

  interval =0:0.01:1
  PyPlot.plot(interval, [value(m2[:a])*x^(power)+(value(m2[:b]) * x) + value(m2[:c]) for x in interval], label = string("ax^", power, " + bx + c"))

  p.set_xlabel("Current density (A/cm²)", fontsize=20)
  p.set_ylabel("Slope", fontsize=20)
  PyPlot.legend(fontsize = 15)
   
  
      PyPlot.scatter(Js, as, label = "Fitted slopes coefficients")
      PyPlot.legend(fontsize = 15)
  

  return value(m2[:a]), value(m2[:b]), value(m2[:c])
end



function get_slope_deg(j, power, a_slope, b_slope, c_slope)
  
  slope = a_slope * j^(power) + b_slope * j + c_slope 
  return slope
  
end




