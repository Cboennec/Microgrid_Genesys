
"""
abstract type AbstractFuelCell <: AbstractConverter  end

An abstract type, parent of all fuel cell types.
"""
abstract type AbstractFuelCell <: AbstractConverter  end


"""
abstract type AbstractFuelCellEffModel end

An abstract type, parent of all fuel cell efficiency model types.
"""
abstract type AbstractFuelCellEffModel end


"""
abstract type AbstractFuelCellAgingModel end

An abstract type, parent of all fuel cell aging model types.
"""
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

# Arrays:
- `powerMax::AbstractArray{Float64,3}`: The maximum power that can be demanded from the fuel cell
- `powerMin::AbstractArray{Float64,3}`: The minimum power that can be demanded from the fuel cell
- `V_J::AbstractArray{Float64,3}`: polarization curve (voltage as a function of current density) for each scenario

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

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell
  powerMin::AbstractArray{Float64,3} #The minimum power that can be demanded to the FuelCell

  V_J::AbstractArray{Float64,3}


  FixedFuelCellEfficiency(;α_p = 0.08,
    η_H2_E = 0.4,
    η_H2_H = 0.4,
    k_aux = 0.15,
    powerMax_ini =.00001,
    ) = new(α_p, η_H2_E, η_H2_H, k_aux, powerMax_ini)
end


"""
PolarizationFuelCellEfficiency

A mutable struct representing a fuel cell efficiency model based on polarization curve.

# Parameters:
- `k_aux::Float64`: Share of the power used by auxiliaries (default: 0.15)
- `K::Float64`: Constant value calculated as Latent Heat Value * molar mass * stoichiometric coefficient / (2 * Faraday constant) (default: (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321))
- `powerMax_ini::Float64`: Initial maximum power (default: 0.00001)
- `V_max::Float64`: Maximum voltage (default: 0.8 V)

# Arrays:
- `powerMax::AbstractArray{Float64,3}`: The maximum power that can be demanded from the fuel cell
- `powerMin::AbstractArray{Float64,3}`: The minimum power that can be demanded from the fuel cell
- `V_J::AbstractArray{Float64,3}`: Polarization curve (voltage as a function of current density) for each scenario

## Example 
```julia
PolarizationFuelCellEfficiency()
```
"""
mutable struct PolarizationFuelCellEfficiency <: AbstractFuelCellEffModel
  k_aux::Float64 # Share of the power used by the auxiliaries
  K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
  powerMax_ini::Float64
  V_max::Float64

  η_H2_E::Float64 #The efficiency from DiHydrogen to Electricity
  η_H2_H::Float64 #The efficiency from DiHydrogen to Heat

  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell
  powerMin::AbstractArray{Float64,3} #The minimum power that can be demanded to the FuelCell

  V_J::AbstractArray{Float64,3}



  PolarizationFuelCellEfficiency(; k_aux = 0.15,
  K = (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321),  #PCI * M_H2 * λ * 3600/(2*F)
  powerMax_ini = .00001,
  V_max = 0.8,
  η_H2_E = 0.4,
  η_H2_H = 0.4
  ) = new(k_aux, K, powerMax_ini, V_max, η_H2_E, η_H2_H)
end

"""
LinearFuelCellEfficiency

A mutable struct representing a linear fuel cell efficiency model.

# Parameters:
- `k_aux::Float64`: Share of the power used by auxiliaries (default: 0.15)
- `K::Float64`: Constant value calculated as Latent Heat Value * molar mass * stoichiometric coefficient / (2 * Faraday constant) (default: (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321))
- `powerMax_ini::Float64`: Initial maximum power (default: 0.00001)
- `V_max::Float64`: Maximum voltage (default: 0.8)
- `a_η::Vector{Float64}`: Slope for the function η(P) 
- `b_η::Vector{Float64}`: Ordinate at the origin for the function η(P) 

# Arrays:
- `powerMax::AbstractArray{Float64,3}`: The maximum power that can be demanded from the fuel cell
- `powerMin::AbstractArray{Float64,3}`: The minimum power that can be demanded from the fuel cell
- `V_J::AbstractArray{Float64,3}`: Polarization curve (voltage as a function of current density) for each scénario

## Example 
```julia
LinearFuelCellEfficiency()
```
"""
mutable struct LinearFuelCellEfficiency <: AbstractFuelCellEffModel
  k_aux::Float64 # Share of the power used by the auxiliaries
  K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
  powerMax_ini::Float64
  V_max::Float64

  η_H2_E::Float64 #The efficiency from DiHydrogen to Electricity
  η_H2_H::Float64 #The efficiency from DiHydrogen to Heat


  powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell
  powerMin::AbstractArray{Float64,3} #The minimum power that can be demanded to the FuelCell

  a_η::Vector{Float64} # the slope for the fucntion η(P)
  b_η::Vector{Float64} # the ordinate at the origin for the function η(P)

  V_J::AbstractArray{Float64,3}


  LinearFuelCellEfficiency(; k_aux = 0.15,
  K = (33.33 *  2.016 * 1.2 * 3600)  / (2*96485.3321),  #PCI * M_H2 * λ * 3600/(2*F)
  powerMax_ini = .00001,
  V_max = 0.8,
  η_H2_E = 0.4,
  η_H2_H = 0.4
  ) = new(k_aux, K, powerMax_ini, V_max, η_H2_E, η_H2_H)
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



"""
`PowerAgingFuelCell` models the aging of a fuel cell based on its polarization curve and operational conditions.

# Fields
- `plot::Bool`: Indicates if degradation plots should be generated.
- `start_stop::Bool`: Enables or disables degradation due to start/stop cycles.
- `update_by_year::Int`: Frequency of State of Health (SoH) and polarization curve updates per year.
- `J_ref::Float64`: Nominal current density of the fuel cell.
- `deg_params::deg_params`: Parameters defining the degradation behavior.
- `V_J_ini::AbstractArray{Float64,2}`: Initial voltage-current relationship at the start of the fuel cell's life.
- `V_J::AbstractArray{Float64,3}`: Time-evolving voltage-current relationship during the fuel cell's life. (with Power as additional dimension)

# Example
```julia
fuel_cell = PowerAgingFuelCell(V_J_ini=initial_data_matrix, objective_hours=15000.0)
```
"""
mutable struct PowerAgingFuelCell <: AbstractFuelCellAgingModel
  plot::Bool 
  start_stop::Bool
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current

  deg_params::deg_params 
  V_J_ini::AbstractArray{Float64,2}

  V_J::AbstractArray{Float64,3}

  function PowerAgingFuelCell(;plot = false,
    start_stop = true,
    update_by_year = 12,
    J_ref = 0.62, 
    V_J_ini = Matrix(transpose(Matrix(DataFrames.DataFrame(CSV.File(joinpath("Examples","data","V_J_PAC.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))))),
    degParams = nothing,
    objective_hours = 15000. #The FuelCell for which we have data is pretty bad so we consider a fuel cell with 15000 hours lifetime at the reference current density.
  ) 

    @assert(objective_hours > 0, "objective_hours should be greater than 0")
    @assert(update_by_year > 0, "update_by_year should be greater than 0")
    @assert(J_ref >= 0, "J_ref should be greater than 0")

    
    #Degradation curves μV/h as a function of the current density for different current densities
    # See (https://oatao.univ-toulouse.fr/29665/1/Pessot_Alexandra.pdf) fig III.43
    P_min = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_min.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
    P_int = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_int.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
    P_max = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_max.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))

    datas_deg_FC = [P_min,P_int,P_max]

    # Coresponding current densities (resp. P_min, P_int, P_max)
    current_densities = [0.075, 0.42, 0.62]

    #The voltage as a function of the current density at the beginning of life
    V_J_FC = Matrix(transpose(Matrix(V_J_ini)))
   
    # Référence current density for this Fuel cell     # See (https://oatao.univ-toulouse.fr/29665/1/Pessot_Alexandra.pdf) 
    J_ref = 0.62

    degParams = create_deg_params(datas_deg_FC, current_densities, V_J_FC, J_ref, objective_hours)

    return new(plot, start_stop, update_by_year, J_ref, degParams, V_J_FC)
  end
end



"""
`FunctHoursAgingFuelCell` models the aging of a fuel cell as a function of its operating hours and current density.

# Fields
- `plot::Bool`: Indicates if degradation plots should be generated.
- `start_stop::Bool`: Enables or disables degradation due to start/stop cycles.
- `deg_params::deg_params`: Parameters defining the degradation behavior based on the polarization curve.
- `update_by_year::Int`: Frequency of State of Health (SoH) and polarization curve updates per year.
- `J_ref::Float64`: Nominal current density of the fuel cell.
- `J_base::Float64`: Current density used as a baseline for degradation calculations.
- `V_J_ini::AbstractArray{Float64,2}`: Initial voltage-current relationship at the start of the fuel cell's life.
- `V_J::AbstractArray{Float64,3}`: Time-evolving voltage-current relationship during the fuel cell's life.
- `coef_a::Float64`: Slope of voltage degradation for each operating hour.
- `coef_b::Float64`: Ordinate at origin of voltage degradation for each operating hour.

# Constructor Parameters
- `plot::Bool=false`: Whether to generate plots for degradation.
- `start_stop::Bool=true`: Whether to include degradation caused by start/stop cycles.
- `deg_params::deg_params=nothing`: Degradation parameters object (optional).
- `update_by_year::Int=12`: Number of updates per year for the SoH and V(I) curves.
- `J_ref::Float64=0.62`: Nominal current density.
- `J_base::Float64=0.1`: Baseline current density for degradation calculations.
- `objective_hours::Float64=15000.0`: Target lifetime in hours at the nominal current density.

# Example
```julia
fuel_cell = FunctHoursAgingFuelCell(
    plot=true,
    start_stop=false,
    J_base=0.1,
    objective_hours=20000.0
)
```
This creates a fuel cell aging model with customized parameters and plots enabled.

"""
mutable struct FunctHoursAgingFuelCell <: AbstractFuelCellAgingModel
  plot::Bool 
  start_stop::Bool
  deg_params::deg_params 
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current density
  J_base::Float64 # The current density used for degradation

  V_J_ini::AbstractArray{Float64,2}
  V_J::AbstractArray{Float64,3}

  coef_a::Float64 #The slope of voltage degradation for each functioning hour
  coef_b::Float64 #The ordinate at origin of voltage degradation for each functioning hour

  function FunctHoursAgingFuelCell(;plot = false,
    start_stop = true,
    deg_params = nothing,
    update_by_year = 12,
    J_ref = 0.62,
    J_base = 0.1, 
    objective_hours = 15000.
  )

  @assert(objective_hours > 0, "objective_hours should be greater than 0")
  @assert(update_by_year > 0, "update_by_year should be greater than 0")
  @assert(J_ref >= 0, "J_ref should be greater than 0")


    #Degradation curves μV/h as a function of the current density for different current densities
    # See (https://oatao.univ-toulouse.fr/29665/1/Pessot_Alexandra.pdf) fig III.43
    P_min = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_min.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
    P_int = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_int.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
    P_max = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_max.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))

    datas_deg_FC = [P_min,P_int,P_max]

    # Coresponding current densities (resp. P_min, P_int, P_max)
    current_densities = [0.075, 0.42, 0.62]

    #The voltage as a function of the current density at the beginning of life
    V_J_FC = Matrix(
      transpose(
        Matrix(
          DataFrames.DataFrame(CSV.File(joinpath("Examples","data","V_J_PAC.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
          )))

    # Référence current density for this Fuel cell     # See (https://oatao.univ-toulouse.fr/29665/1/Pessot_Alexandra.pdf) 
    J_ref = 0.62

    degParams = create_deg_params(datas_deg_FC, current_densities, V_J_FC, J_ref, objective_hours)

    return new(plot, start_stop, degParams, update_by_year, J_ref, J_base)
  end
end



"""
`FixedLifetimeFuelCell` models the aging of a fuel cell with a fixed operational lifetime.

# Fields
- `plot::Bool`: Indicates if degradation plots should be generated.
- `update_by_year::Int`: Frequency of State of Health (SoH) and polarization curve updates per year.
- `J_ref::Float64`: Nominal current density of the fuel cell.
- `nHourMax::Int64`: Maximum number of operational hours.
- `lifetime::Int64`: Fixed lifetime of the fuel cell in years (derived from `nHourMax`).
- `V_J_ini::AbstractArray{Float64,2}`: Initial voltage-current relationship at the start of the fuel cell's life.
- `V_J::AbstractArray{Float64,3}`: Time-evolving voltage-current relationship during the fuel cell's life.
- `V_nom_ini::Float64`: Nominal voltage at the start of the fuel cell's life (not set in the constructor).

# Constructor Parameters
- `plot::Bool=false`: Whether to generate plots for degradation.
- `update_by_year::Int=12`: Number of updates per year for the SoH and V(I) curves.
- `J_ref::Float64=0.62`: Nominal current density.
- `nHourMax::Int64=87600`: Maximum operational hours (default: 10 years).

# Example
```julia
fuel_cell = FixedLifetimeFuelCell(plot=true, update_by_year=6, J_ref=0.7, nHourMax=43800)
```
This creates a fuel cell aging model with a lifetime of 5 years, using the provided nHourMax
"""
mutable struct FixedLifetimeFuelCell <: AbstractFuelCellAgingModel
  plot::Bool 
  update_by_year::Int # The frequency (per year) of SoH and V(I) update
  J_ref::Float64 # The nominal current density
  nHourMax::Int64
  lifetime::Float64

  V_J_ini::AbstractArray{Float64,2}
  V_J::AbstractArray{Float64,3}

  V_nom_ini::Float64

  function FixedLifetimeFuelCell(;plot = false,
    update_by_year = 12,
    J_ref = 0.62,
    nHourMax = 87600
  ) 
    lifetime = div(nHourMax, 8760) 
    new(plot, update_by_year, J_ref, nHourMax, lifetime)
  end
end



"""
`FuelCell` represents the main structure for modeling fuel cell behavior, including efficiency, aging, and other key parameters.

# Fields
- `eff_model::AbstractFuelCellEffModel`: The efficiency model used to calculate fuel cell efficiency.
- `SoH_model::AbstractFuelCellAgingModel`: The aging model used to track the State of Health (SoH) of the fuel cell over time.
- `couplage::Bool`: Indicates whether the SoH should influence other parameters (e.g., efficiency, output voltage).
- `bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}`: Bounds for operational parameters, such as voltage or power.
- `SoH_threshold::Float64`: SoH level at which the fuel cell should be replaced.
- `V_J_ini::AbstractArray{Float64,2}`: Initial voltage-current relationship for the fuel cell.
- `soh_ini::Float64`: Initial SoH at the start of the simulation (default: 1.0, corresponding to 100% health).
- `N_cell::Int64`: Number of cells assembled in the fuel cell stack.
- `surface::Float64`: Surface area of the fuel cell stack (unit should be documented if relevant).
- `η::AbstractArray{Float64,3}`: Efficiency evolution over time (hydrogen to electricity).
- `carrier::Vector{EnergyCarrier}`: Vector of energy carriers used by the fuel cell (e.g., hydrogen, oxygen).
- `soh::AbstractArray{Float64,3}`: Time-evolving State of Health for the fuel cell.
- `cost::AbstractArray{Float64,2}`: Economic metrics associated with fuel cell operation (e.g., cost per hour, maintenance).

# Constructor Parameters
- `eff_model::AbstractFuelCellEffModel=PolarizationFuelCellEfficiency()`: Efficiency model to be used.
- `SoH_model::AbstractFuelCellAgingModel=PowerAgingFuelCell()`: Aging model to be used.
- `couplage::Bool=true`: Whether SoH influences other parameters.
- `bounds=(lb=0.0, ub=50.0)`: sizing power bounds for all-in-one design.
- `SoH_threshold::Float64=0.9`: SoH replacement threshold (default: 90%).
- `V_J_ini::AbstractArray{Float64,2}`: Initial voltage-current data (default loaded from a CSV file).
- `soh_ini::Float64=1.0`: Initial SoH at the start of the simulation.

# Example
```julia
fuel_cell = FuelCell(eff_model=PolarizationFuelCellEfficiency(),
                     SoH_model=PowerAgingFuelCell(),
                     couplage=true,
                     bounds=(lb=0.0, ub=60.0),
                     SoH_threshold=0.8,
                     soh_ini=1.0)
```
This initializes a FuelCell structure with custom sizing power bounds and a lower SoH replacement threshold.
"""
mutable struct FuelCell <: AbstractFuelCell

  eff_model::AbstractFuelCellEffModel
	SoH_model::AbstractFuelCellAgingModel
  couplage::Bool  #a boolean tuple to tell wether or not the soh should influence the other parameters.

	
	bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
	SoH_threshold::Float64 # SoH level to replace battery
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

	FuelCell(;eff_model = PolarizationFuelCellEfficiency(),
    SoH_model = PowerAgingFuelCell(),
    couplage = true,
    bounds = (lb = 0., ub = 50.),
    SoH_threshold = 0.9,
    V_J_ini = Matrix(transpose(Matrix(DataFrames.DataFrame(CSV.File(joinpath("Examples","data","V_J_PAC.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))))),
    soh_ini = 1. 
  ) = new(eff_model, SoH_model, couplage, bounds, SoH_threshold, V_J_ini, soh_ini)

end



  
  ### Preallocation
  function preallocate!(fc::FuelCell, nh::Int64, ny::Int64, ns::Int64)

    fc.eff_model.powerMax = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  fc.eff_model.powerMax[1,1,:] .= fc.eff_model.powerMax_ini
    fc.eff_model.powerMin = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  fc.eff_model.powerMin[1,1,:] .= fc.eff_model.powerMax_ini
    fc.η = convert(SharedArray,zeros(nh+1, ny+1, ns))
    fc.carrier = [Electricity(), Heat(), Hydrogen()]
    fc.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
    fc.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; fc.soh[1,1,:] .= fc.soh_ini
    fc.cost = convert(SharedArray,zeros(ny, ns))



    fc.SoH_model.V_J_ini = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P
    fc.SoH_model.V_J = zeros(3, length(fc.V_J_ini[1,:]), ns) #J, V, P
    fc.eff_model.V_J = zeros(3, length(fc.V_J_ini[1,:]), ns) #J, V, P

    
    if fc.eff_model isa LinearFuelCellEfficiency
      fc.eff_model.a_η =  convert(SharedArray, zeros(ns))
      fc.eff_model.b_η = convert(SharedArray, zeros(ns))
    end


    return fc
end
  

### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, fc::FuelCell, decision::Float64, Δh::Int64)

  fc.carrier[1].power[h,y,s], fc.carrier[2].power[h,y,s], fc.carrier[3].power[h,y,s] = compute_operation_efficiency(fc, fc.eff_model, h ,y ,s , decision, Δh)

  fc.soh[h+1,y,s], fc.eff_model.powerMax[h+1,y,s], fc.eff_model.powerMin[h+1,y,s] = compute_operation_soh(fc, fc.SoH_model, h ,y ,s, Δh)

end


### Operation dynamic
function compute_operation_dynamics(fc::FuelCell, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)

  return compute_operation_efficiency(fc, fc.eff_model, h ,y ,s , decision, Δh)

end


function compute_operation_efficiency(fc::FuelCell, model::PolarizationFuelCellEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
 #Apply minimum power
  model.powerMin[h,y,s] <= decision ? power_E = min(decision, model.powerMax[h,y,s]) : power_E = 0. 
      
  if power_E > 0
      #Compute the power needed to feed the auxiliaries and deliver the right power
      P_tot = floor(power_E / (1 - model.k_aux); digits=6)

      #Find the corresponding current from an interpolation from P(I) curve 
      j = interpolation(model.V_J[3,:,s], model.V_J[1,:,s], P_tot, true)
     
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
 
      η_E = model.a_η[s] * P_tot + model.b_η[s]
      
      if η_E >= 0.46
        println("y,h = ", y, ", ", h, "   a_η = ", model.a_η[s], " , P_tot = ", P_tot, ", b_η = ",  model.b_η[s] )
        println("configuration = ",typeof(fc.eff_model) , ", ", typeof(fc.SoH_model), ", surface = ", fc.surface, ", Ncell = ", fc.N_cell, ", couple = ", fc.couplage )
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
          if p > maximum(fc.eff_model.V_J[3,:,s])
            println("Power = ", p, " > ", "max P(J) = ", maximum(fc.eff_model.V_J[3,:,s]), " h, y, s = ", h, ", " , y, ", ", s)
            println("configuration = ",typeof(fc.eff_model) , ", ", typeof(fc.SoH_model), ", surface = ", fc.surface, ", Ncell = ", fc.N_cell, ", couplage = ", fc.couplage )

            p = floor(maximum(model.V_J[3,:,s]), digits =2)
          end
            push!(current_densities, interpolation(fc.eff_model.V_J[3,:,s], fc.eff_model.V_J[1,:,s], p, true))
        end
    end

    for j in current_densities
       coef_a = get_slope_deg(j, model.deg_params.power_slope, model.deg_params.a_slope, model.deg_params.b_slope, model.deg_params.c_slope)
 
       ΔV = (model.V_J[1,:,s] * coef_a).+coef_b 

       ΔV *= model.deg_params.adjustment_coef 

       #Adjust with time spent (ref time is 1 hour)
       ΔV *= Δh

       model.V_J[2,:,s] .-= ΔV
    end


    V_nom = interpolation(model.V_J[1,:,s], model.V_J[2,:,s], model.J_ref, true)

    if model.start_stop
      start_stop_count = get_start_stops(powers)
      model.V_J[2,:,s] .-= model.deg_params.start_stop_coef * V_nom * start_stop_count
    end 

    model.V_J[3,:,s] = model.V_J[2,:,s] .* (model.V_J[1,:,s] * fc.surface * fc.N_cell) 

    V_nom_ini = interpolation(model.V_J_ini[1,:], model.V_J_ini[2,:], model.J_ref, true)
    V_nom = interpolation(model.V_J[1,:,s], model.V_J[2,:,s], model.J_ref, true)


    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:,s], model.V_J[2,:,s])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end

    nextSoH = V_nom/V_nom_ini  

    if fc.couplage
      nextPowerMax = maximum(model.V_J[3,:,s]) * (1-fc.eff_model.k_aux)
      nextPowerMin = compute_min_power(fc, s)
      fc.eff_model.V_J[:,:,s] = copy(model.V_J[:,:,s])

      if fc.eff_model isa LinearFuelCellEfficiency 
        update_η_lin(fc, fc.eff_model, s)
      end

    else
      nextPowerMax = fc.eff_model.powerMax[h,y,s] 
      nextPowerMin = fc.eff_model.powerMin[h,y,s] 
    end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.eff_model.powerMax[h,y,s] 
    nextPowerMin = fc.eff_model.powerMin[h,y,s] 
  end

  return nextSoH, nextPowerMax, nextPowerMin
end


function compute_operation_soh(fc::FuelCell, model::FunctHoursAgingFuelCell, h::Int64,  y::Int64,  s::Int64,  Δh::Int64)

  if (h%convert(Int64,floor(8760/model.update_by_year))) == 0 
    interval = (h-convert(Int64,floor(8760/model.update_by_year))+1):h
    
    powers = fc.carrier[1].power[interval,y,s]

    n_hours_active = sum(powers .> 1e-6)

    #Base degradation considered at nominal current density
    ΔV = (model.V_J[1,:,s] * model.coef_a) .+ model.coef_b

    ΔV *= model.deg_params.adjustment_coef

    #Adjust with time spent (ref time is 1 hour)
    ΔV *= n_hours_active * Δh
       
    model.V_J[2,:,s] .-= ΔV
  
  

    V_nom = interpolation(model.V_J[1,:,s], model.V_J[2,:,s], model.J_ref, true)

    if model.start_stop
      start_stop_count = get_start_stops(powers)
      model.V_J[2,:,s] .-= model.deg_params.start_stop_coef * V_nom * start_stop_count
    end 

    model.V_J[3,:,s] = model.V_J[2,:,s] .* (model.V_J[1,:,s] * fc.surface * fc.N_cell) 

    V_nom_ini = interpolation(model.V_J_ini[1,:], model.V_J_ini[2,:], model.J_ref, true)
    V_nom = interpolation(model.V_J[1,:,s], model.V_J[2,:,s], model.J_ref, true)


    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:,s], model.V_J[2,:,s])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end

    nextSoH = V_nom/V_nom_ini  

      if fc.couplage
        nextPowerMax = maximum(model.V_J[3,:,s]) * (1-fc.eff_model.k_aux)
        nextPowerMin = compute_min_power(fc, s)
        fc.eff_model.V_J[:,:,s] = copy(model.V_J[:,:,s])

          if fc.eff_model isa LinearFuelCellEfficiency 
            update_η_lin(fc, fc.eff_model, s)
          end

      else
        nextPowerMax = fc.eff_model.powerMax[h,y,s] 
        nextPowerMin = fc.eff_model.powerMin[h,y,s] 
      end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.eff_model.powerMax[h,y,s] 
    nextPowerMin = fc.eff_model.powerMin[h,y,s] 
  end

  return nextSoH, nextPowerMax, nextPowerMin
end


function compute_operation_soh(fc::FuelCell, model::FixedLifetimeFuelCell, h::Int64,  y::Int64,  s::Int64,  Δh::Int64)

  if (h%convert(Int64,floor(8760/model.update_by_year))) == 0 
    duration = convert(Int64,floor(8760/model.update_by_year))
    
    frac = duration/model.nHourMax
    #Base degradation considered at nominal current density
  
    ΔV = frac * model.V_nom_ini * (1-fc.SoH_threshold)
       
    model.V_J[2,:,s] .-= ΔV
    
    model.V_J[3,:,s] = model.V_J[2,:,s] .* (model.V_J[1,:,s] * fc.surface * fc.N_cell) 

    nextSoH = fc.soh[h,y,s] - (frac * (1-fc.SoH_threshold))

    if model.plot
      plt = PyPlot.subplot()
      PyPlot.plot(model.V_J[1,:,s], model.V_J[2,:,s])
      plt.set_ylabel("Tension (V)")
      plt.set_xlabel("Current density (A/cm²)")
    end


      if fc.couplage
        nextPowerMax = maximum(model.V_J[3,:,s]) * (1-fc.eff_model.k_aux)
        nextPowerMin = compute_min_power(fc, s)
        fc.eff_model.V_J[:,:,s] = copy(model.V_J[:,:,s])

          if fc.eff_model isa LinearFuelCellEfficiency 
            update_η_lin(fc, fc.eff_model, s)
          end

      else
        nextPowerMax = fc.eff_model.powerMax[h,y,s] 
        nextPowerMin = fc.eff_model.powerMin[h,y,s] 
      end
  else 
    nextSoH = fc.soh[h,y,s] 
    nextPowerMax = fc.eff_model.powerMax[h,y,s] 
    nextPowerMin = fc.eff_model.powerMin[h,y,s] 
  end

  return nextSoH, nextPowerMax, nextPowerMin
end


function initialize_investments!(s::Int64, fc::FuelCell, decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})

  fc.surface = decision.surface
  fc.N_cell = decision.N_cell
  fc.soh[1,1,s] = fc.soh_ini


  fc.eff_model.V_J[1,:,s] = copy(fc.V_J_ini[1,:]) 
  fc.eff_model.V_J[2,:,s] = copy(fc.V_J_ini[2,:])
  fc.eff_model.V_J[3,:,s] = fc.V_J_ini[2,:] .* fc.V_J_ini[1,:] * fc.surface * fc.N_cell
  fc.eff_model.powerMax[1,1,s] = maximum(fc.eff_model.V_J[3,:,s]) * (1-fc.eff_model.k_aux)
  fc.eff_model.powerMin[1,1,s] = compute_min_power(fc, s)

  if fc.eff_model isa LinearFuelCellEfficiency 
    update_η_lin(fc, fc.eff_model, s)
  end

  #Initialization of V(J)
    fc.SoH_model.V_J_ini[1,:] = copy(fc.V_J_ini[1,:]) 
    fc.SoH_model.V_J_ini[2,:] = copy(fc.V_J_ini[2,:])
    fc.SoH_model.V_J_ini[3,:] = fc.V_J_ini[2,:] .* fc.V_J_ini[1,:] * fc.surface * fc.N_cell
    fc.SoH_model.V_J .= copy(fc.SoH_model.V_J_ini)

  if fc.SoH_model isa FunctHoursAgingFuelCell

    fc.SoH_model.coef_b = fc.SoH_model.deg_params.b
    fc.SoH_model.coef_a = get_slope_deg(fc.SoH_model.J_base, fc.SoH_model.deg_params.power_slope, fc.SoH_model.deg_params.a_slope, fc.SoH_model.deg_params.b_slope, fc.SoH_model.deg_params.c_slope)
  
  elseif fc.SoH_model isa FixedLifetimeFuelCell

    fc.SoH_model.V_nom_ini = interpolation(fc.V_J_ini[1,:], fc.V_J_ini[2,:], fc.SoH_model.J_ref , true)

  end

  
end




  ### Investment dynamic
  function compute_investment_dynamics!(y::Int64, s::Int64, fc::FuelCell,  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}})    
    fc.eff_model.powerMax[1,y+1,s], fc.eff_model.powerMin[1,y+1,s], fc.soh[1,y+1,s] = compute_investment_dynamics(fc, (powerMax = fc.eff_model.powerMax[end,y,s], powerMin = fc.eff_model.powerMin[end,y,s], soh = fc.soh[end,y,s]), decision, s)
  end

  
  function compute_investment_dynamics(fc::FuelCell, state::NamedTuple{(:powerMax, :powerMin, :soh), Tuple{Float64, Float64, Float64}},  decision::NamedTuple{(:surface, :N_cell), Tuple{Float64, Int64}}, s::Int64)
    if decision.N_cell > 1e-2 

        V_J = zeros(3, length(fc.V_J_ini[1,:])) #J, V, P

        fc.surface = decision.surface
        fc.N_cell = decision.N_cell
        

        for (i,a) in enumerate([fc.V_J_ini[1,:], fc.V_J_ini[2,:], fc.V_J_ini[2,:] .* fc.V_J_ini[1,:] * fc.surface * fc.N_cell])
            V_J[i,:] = copy(a) 
        end

        fc.SoH_model.J_ref = 0.62

        soh_next = fc.soh_ini

        fc.SoH_model.V_J[:,:,s] = copy(V_J)
        fc.eff_model.V_J[:,:,s] = copy(V_J)

        powerMax_next = maximum(V_J[3,:]) * (1-fc.eff_model.k_aux)

        powerMin_next = compute_min_power(fc, s)

        if fc.eff_model isa LinearFuelCellEfficiency 
          update_η_lin(fc, fc.eff_model, s)
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



function get_η_E(P_net::Float64, fc::FuelCell, s::Int64)

  
  P_tot = floor(P_net / (1 - fc.eff_model.k_aux); digits=6)

  
  #Find the corresponding current from an interpolation from P(I) curve 
  j = interpolation(fc.eff_model.V_J[3,:,s], fc.eff_model.V_J[1,:,s], P_tot, true )
  i = j * fc.surface

  return P_net / (fc.eff_model.K * i * fc.N_cell)

end




#compute the power that correpond to the maximum allowed tension
function compute_min_power(fc::FuelCell, s::Int64)
  if fc.eff_model isa FixedFuelCellEfficiency
    P_min = fc.eff_model.α_p * maximum(fc.eff_model.V_J[3,:,s])
  else
    P_min_tot = interpolation(fc.eff_model.V_J[2,:,s], fc.eff_model.V_J[3,:,s], fc.eff_model.V_max, false )
    P_min = P_min_tot / (1 + fc.eff_model.k_aux)
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


function update_η_lin(fc::FuelCell, model::LinearFuelCellEfficiency, s::Int64)

    
  P_max = maximum(model.V_J[3,:,s]) * (1-model.k_aux)
  P_min = compute_min_power(fc, s)


  if P_max == 0.
    η_P_min, η_P_max = 0., 0.
  else
    η_P_min = get_η_E(P_min, fc, s)
    η_P_max = get_η_E(P_max, fc, s)
  end
  

  a_η = (η_P_max - η_P_min) / (P_max - P_min)
  b_η = η_P_min - a_η * P_min
 


  fc.eff_model.a_η[s] = a_η
  fc.eff_model.b_η[s] = b_η

end


function toStringShort(fc::FuelCell)

	if fc.eff_model isa LinearFuelCellEfficiency
		efficiency = "x"
	elseif fc.eff_model isa PolarizationFuelCellEfficiency
		efficiency = "V(J)"
  elseif fc.eff_model isa FixedFuelCellEfficiency
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



function create_deg_params(datas::Vector{DataFrames.DataFrame}, Js::Vector{Float64}, V_J::Matrix{Float64}, J_ref::Float64, objective_lifetime::Float64; power = 1/2, plot = false, start_stop_coef = 0.0000196)

  #Maximum deg profil in Alexandra Pessot thesis
  P_max = datas[3]

  #Get affine coef from input data
  as, b = fit_all_curves(datas, Js; plot = plot)

  
  #get a,b,c coef to be able to write the degradation as ax^power + bx + c
  a_slope, b_slope, c_slope = fit_dot(Js, as, power; plot = plot)

  #initial voltage at ref current density
  V_ini_ref =  interpolation(V_J[1,:],  V_J[2,:], J_ref, true)

  ΔV_tot = V_ini_ref * 0.1

  #Voltage ref loss
  a = a_slope * (J_ref ^ power) + b_slope * J_ref + c_slope
  V_deg_ref = a * J_ref + b
  V_deg_ref = interpolation(P_max.J,  P_max.V, J_ref, true)

  #Adujst the lifetime to fit a target (The FC will be able to be used at its ref current density for the target amount of hour)
  current_lifetime = ΔV_tot / (V_deg_ref * 1e-6)

  ###########################################################################################
  ######################### Attention !!!!!    ##########################################
  ###########################################################################################
  adaptation_coefficient = current_lifetime * 1e-6/objective_lifetime # Attention à ce 1e-6 qui sert à mettre les deg en microvolt et ne devrait pas être ici

  start_stop_coef = start_stop_coef #0.00196% as stated in Pucheng Pei, Qianfei Chang, Tian Tang,
 # A quick evaluating method for automotive fuel cell lifetime (https://doi.org/10.1016/j.ijhydene.2008.04.048)
  
  return deg_params(a_slope, b_slope, c_slope, power, b, adaptation_coefficient, start_stop_coef)

end



   
  


function fit_all_curves(data, Js; plot = false, silent = true)

  n_data = length(data)
  n_data_point = [length(data[i].J) for i in 1:n_data]


  m2 = Model(Gurobi.Optimizer)
  if silent
    set_optimizer_attribute(m2, "OutputFlag", 0)
  end
  
  @variable(m2, a[1:n_data] >= 0)
  @variable(m2, b >= 0)

  @variable(m2, errors[1:n_data])
  @constraint(m2, [d in 1:n_data], errors[d] >= sum( (a[d]*data[d].J[i]+b - data[d].V[i])^2 for i in 1:n_data_point[d]))

  #Minimize the squared error
  @objective(m2, Min, sum(errors[d] for d in 1:n_data))


  JuMP.optimize!(m2)
  if plot
    plt.rcParams["text.usetex"] = true

    fig, ax = plt.subplots()
    fig.set_size_inches( 1920 / fig.dpi, 1080/ fig.dpi)

    for d in 1:n_data
      PyPlot.scatter(data[d].J, data[d].V, label = string(Js[d] , " [A/cm²] : data"))
      PyPlot.plot(data[d].J, data[d].J .* value.(m2[:a])[d] .+ value(m2[:b]), label = string(Js[d] , " [A/cm²] : model"))
    end
    PyPlot.xlabel("Current density [A/cm²]")
    PyPlot.ylabel("\$ΔV\$ [μV/h]")
    PyPlot.legend()

    PyPlot.tight_layout()

  end


  return value.(m2[:a]), value(m2[:b])
end


function fit_dot(Js, as, power; plot = false, silent = true)

  n_data_point = length(Js)

  m2 = Model(Gurobi.Optimizer)
  if silent
    set_optimizer_attribute(m2, "OutputFlag", 0)
  end
  
  @variable(m2, a)
  @variable(m2, b)
  @variable(m2, c)


  @variable(m2, errors[1:n_data_point])
  @constraint(m2, [i in 1:n_data_point], errors[i] == (a*Js[i]^(power)+(b * Js[i]) + c) - as[i])

  #Minimize the squared error
  @objective(m2, Min, sum((errors[i]^2) for i in 1:n_data_point))


  JuMP.optimize!(m2)

  if plot
    fig, ax = plt.subplots()
    fig.set_size_inches( 1920 / fig.dpi, 1080/ fig.dpi)

    interval = 0:0.01:1
    PyPlot.plot(interval, [value(m2[:a])*x^(power)+(value(m2[:b]) * x) + value(m2[:c]) for x in interval], label = string("\$ax^{$power} + bx + c\$"))

    PyPlot.xlabel("Current density [A/cm²]")
    PyPlot.ylabel("Slope")
    PyPlot.legend()
    plt.rcParams["text.usetex"] = true

    PyPlot.vlines(0.14, ymin = ax.get_ylim()[1], ymax = ax.get_ylim()[2] , color = "black", linewidth = 2)
    ax[:set_ylim]([0,ax.get_ylim()[2]])
    ax[:set_xlim]([0,ax.get_xlim()[2]])

    
    PyPlot.scatter(Js, as, label = "Fitted slopes coefficients")
    PyPlot.legend()

    PyPlot.tight_layout()
  end

  return value(m2[:a]), value(m2[:b]), value(m2[:c])
end



function get_slope_deg(j, power, a_slope, b_slope, c_slope)
  
  slope = a_slope * j^(power) + b_slope * j + c_slope 
  return slope
  
end




