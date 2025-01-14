abstract type AbstractLiion <: AbstractStorage  end

abstract type AbstractLiionEffModel end

abstract type AbstractLiionAgingModel end

"""
FixedLiionEfficiency <: AbstractLiionEffModel

A mutable struct that represents a Li-ion battery efficiency model for SoC computation.
This model implement a fixed efficiency that can decrease linearly with battery aging.
	
# Parameters:
- `η_ch::Float64`: Charging efficiency (default : 0.98)
- `η_dch::Float64`: Discharging efficiency (default : 0.98)
- `α_p_ch::Float64`: Maximum charging C-rate (default : 1.5)
- `α_p_dch::Float64`: Maximum discharging C-rate (default : 1.5)
- `η_deg_coef::Float64`: The efficiency degradation coefficient (default : 0.2303,  ref : Redondo Iglesias - Efficiency Degradation Model of Lithium-Ion Batteries for Electric Vehicles)
- `couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}  `: Named tuple with two boolean values to indicate if the SoH should influence the other parameters (E stand for capacity coupling and R for efficiency coupling) (default : (E = true, R = true))
- `η_self::Float64`: Named tuple with two boolean values to indicate if the SoH should influence the other parameters (E stand for capacity coupling and R for efficiency coupling) (default : 0.0005)

## Example 
```julia
FixedLiionEfficiency()
```
"""
mutable struct FixedLiionEfficiency <: AbstractLiionEffModel

	η_ch::Float64 #Charging efficiency
	η_dch::Float64 #Discharging efficiency
	η_deg_coef::Float64 #The efficiency degradation coefficient
	couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}  #a boolean tuple to tell wether or not the soh should influence the other parameters.
	α_p_ch::Float64 #C_rate max
	α_p_dch::Float64 #C_rate max
	η_self::Float64 #Auto discarge factor
	
	
	FixedLiionEfficiency(;η_ch = 0.98,
		η_dch = 0.98,
		η_deg_coef = 0.2303,  # ref : Redondo Iglesias - Efficiency Degradation Model of Lithium-Ion Batteries for Electric Vehicles
		couplage = (E = true, R = true),
		α_p_ch = 1.5,
		α_p_dch = 1.5,
		η_self = 0.0005,
		) = new(η_ch, η_dch, η_deg_coef, couplage, α_p_ch, α_p_dch ,η_self)

end

"""
PolynomialLiionEfficiency <: AbstractLiionEffModel

A mutable struct that represents a Li-ion battery efficiency model with polynomial efficiency characteristics for SoC computation.
This model implements an efficiency model based on polynomial coefficients of the form `ax² + bx + c`  that can be used to calculate efficiency.

# Parameters:
- `a_η_ch::Float64`: Coefficient 'a' for charging efficiency (default: 0.0033)
- `b_η_ch::Float64`: Coefficient 'b' for charging efficiency (default: 0.0297)
- `c_η_ch::Float64`: Coefficient 'c' for charging efficiency (default: 0.99814)
- `a_η_dch::Float64`: Coefficient 'a' for discharging efficiency (default: 0.002232)
- `b_η_dch::Float64`: Coefficient 'b' for discharging efficiency (default: 0.0246)
- `c_η_dch::Float64`: Coefficient 'c' for discharging efficiency (default: 1)
- `η_deg_coef::Float64`: The efficiency degradation coefficient (default: 0.2303, ref: Redondo Iglesias - Efficiency Degradation Model of Lithium-Ion Batteries for Electric Vehicles)
- `couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}`: Named tuple with two boolean values to indicate if the SoH should influence the other parameters (E stands for capacity coupling and R for efficiency coupling) (default: (E = true, R = false))
- `α_p_ch::Float64`: Maximum charging C-rate (default: 1.5)
- `α_p_dch::Float64`: Maximum discharging C-rate (default: 1.5)
- `η_self::Float64`: Auto-discharge factor (default: 0.0005)

## Example 
```julia
PolynomialLiionEfficiency()
```
"""
mutable struct PolynomialLiionEfficiency <: AbstractLiionEffModel

	η_ch::Float64 #Charging efficiency
	η_dch::Float64 #Discharging efficiency

	#Polynom coefficients for the computation of efficiency
	a_η_ch::Float64
	b_η_ch::Float64
	c_η_ch::Float64
	a_η_dch::Float64
	b_η_dch::Float64
	c_η_dch::Float64

	η_deg_coef::Float64 #The efficiency degradation coefficient
	couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}  #a boolean tuple to tell wether or not the soh should influence the other parameters.
	α_p_ch::Float64 #C_rate max
	α_p_dch::Float64 #C_rate max
	η_self::Float64 #Auto discarge factor 
	
	
	PolynomialLiionEfficiency(;
	η_ch = 0.98,
	η_dch = 0.98,
	a_η_ch=	0.0033,
	b_η_ch = 0.0297, 
	c_η_ch = 0.99814,
	a_η_dch = 0.002232, 
	b_η_dch = 0.0246, 
	c_η_dch = 1,
	η_deg_coef = 0.2303, # ref : Redondo Iglesias - Efficiency Degradation Model of Lithium-Ion Batteries for Electric Vehicles
	couplage = (E = true, R = false),
	α_p_ch = 1.5,
	α_p_dch = 1.5,
	η_self = 0.0005) = new(η_ch, η_dch, a_η_ch, b_η_ch, c_η_ch, a_η_dch, b_η_dch, c_η_dch, η_deg_coef, couplage, α_p_ch, α_p_dch ,η_self)

end


"""
EnergyThroughputLiion <: AbstractLiionAgingModel

A mutable struct that represents an aging model for Li-ion batteries based on energy throughput, which accounts for calendar aging and cycling aging.
This model calculates aging based on the cumulative energy throughput and additionnaly calendar aging.

# Parameters:
- `calendar::Bool`: A boolean value to indicate whether to consider calendar aging (default: true)
- `nCycle::Int64`: An adjusted number of cycle to reach the SoH replacement threshold after nCycle_ini cycles (default: 6000)
- `nCycle_ini::Int64`: The total number of cycles achievable before reaching EOL(default: 6000). 
- `Δcal::Float64`: The calendar aging parameter (default: 1 - exp(-4.14e-10 * 3600))

## Example 
```julia
EnergyThroughputLiion(;nCycle_ini = Int(floor(fatigue_data.cycle[findfirst(fatigue_data.DoD .> (0.6))])))```

This set the feasible number of cycles corresponding to the number of 60% DoD cycles in the cycle-to-failure curve (named fatigue_data).
Supposing that fatigue data is a dataframe with a cycle column and a DoD column
"""
mutable struct EnergyThroughputLiion <: AbstractLiionAgingModel

	calendar::Bool
	nCycle::Int64
	nCycle_ini::Int64
	Δcal::Float64 

	EnergyThroughputLiion(;calendar = true,
	nCycle = 2500.,
	nCycle_ini = 2500.,
	Δcal = (1 - exp(- 4.14e-10 * 3600))
	) = new(calendar, nCycle, nCycle_ini, Δcal)
end

"""
FixedLifetimeLiion <: AbstractLiionAgingModel

A mutable struct that represents an aging model for Li-ion batteries with a fixed lifetime in years.

# Parameters:
- `lifetime::Int64`: The fixed lifetime of the battery in years (default: 12)

## Example 
```julia
FixedLifetimeLiion()
```
"""
mutable struct FixedLifetimeLiion <: AbstractLiionAgingModel

	lifetime::Int64

	FixedLifetimeLiion(;lifetime = 12) = new(lifetime)
end


"""
RainflowLiion <: AbstractLiionAgingModel

A mutable struct that represents an aging model for Li-ion batteries based on rainflow cycle counting, considering calendar aging and cycling aging.

# Parameters:
- `calendar::Bool`: A boolean value for the activation of calendar aging (default: true)
- `update_by_year::Int64`: Number of updates each year (default: 12)
- `fatigue_data::DataFrames.DataFrame`: Cycle to failure curve data (default: loaded from a CSV file with a DoD column and a nCycle column)

## Example 
```julia
RainflowLiion(fatigue_data = fatigue_data)```

supposing that fatigue_data is a dataframe with a cycle column and a DoD column
"""
mutable struct RainflowLiion <: AbstractLiionAgingModel
	calendar::Bool#Booleann for the activation of calendar aging

	update_by_year::Int64 #Number of update each year

	fatigue_data::DataFrames.DataFrame #Cycle to failure curve 

	RainflowLiion(;calendar = true,
	update_by_year = 12,
	fatigue_data = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","fatigue_data_NMC.csv"), delim = ",", header = [Symbol("DoD"),Symbol("cycle")], types=Dict(:DoD=>Float64, :cycle=>Float64)))
	) = new(calendar, update_by_year, fatigue_data)

end

"""
SemiEmpiricalLiion <: AbstractLiionAgingModel

A mutable struct that represents an aging model for Li-ion batteries based on a semi-empirical model, accounting for temperature and other parameters.

# Parameters:
- `update_by_year::Int64`: Number of updates each year (default: 12)
- Parameters for NMC bateries (as in Technoeconomic model of second-life batteries for utility-scale solar considering calendar and cycle aging Ian Mathews, Bolun Xu, Wei He, Vanessa Barreto, Tonio Buonassisi, Ian Marius Peters):
  - `alpha_sei::Float64`
  - `beta_sei::Float64`
  - `k_delta1::Float64`
  - `k_delta2::Float64`
  - `k_sigma::Float64`
  - `sigma_ref::Float64`
  - `k_T::Float64`
  - `T_ref::Float64`
  - `k_t::Float64`
- `temperature::Float64`: Temperature of the battery in Kelvin (default: 298)
- `Sum_fd::AbstractArray{Float64,1}`: Used as a memory of the cumulated fatigue of the battery

## Example 
```julia
SemiEmpiricalLiion()
```
"""
mutable struct SemiEmpiricalLiion <: AbstractLiionAgingModel
	update_by_year::Int64 #Number of update each year
	
	# for NMC parameters #
   #Technoeconomic model of second-life batteries for utility-scale solar
   #considering calendar and cycle aging
   #Ian Mathews a,⁎, Bolun Xu b, Wei He a, Vanessa Barreto c, Tonio Buonassisi a, Ian Marius Peters

	alpha_sei::Float64
	beta_sei::Float64
	k_delta1::Float64
	k_delta2::Float64
	k_sigma::Float64
	sigma_ref::Float64
	k_T::Float64 
	T_ref::Float64
	k_t::Float64

	temperature::Float64 #temprature of the battery

	Sum_fd::AbstractArray{Float64,1} # This is used as a memory of the cumulated fatigue of the battery, see ref above for some details

	SemiEmpiricalLiion(;update_by_year = 12,
	alpha_sei = 5.75e-2,
    beta_sei = 121,
    k_delta1 = 1.0487e-4,
    k_delta2 = 2.03,
    k_sigma = 1.04,
    sigma_ref = 0.5,
   	k_T = 6.93e-2,
   	T_ref = 298,
   	k_t = 4.14e-10,
	temperature = 298) = new(update_by_year, alpha_sei, beta_sei, k_delta1, k_delta2, k_sigma, sigma_ref, k_T, T_ref, k_t, temperature)
end

"""
# Liion

A mutable struct representing a Li-ion battery model with state of charge (SoC) computation and aging models.

## Parameters

- `eff_model::AbstractLiionEffModel`: Model for state of charge computation.
- `SoH_model::AbstractLiionAgingModel`: Model for aging computation.
- `α_soc_min::Float64`: Minimum threshold of charge (normalized).
- `α_soc_max::Float64`: Maximum threshold of charge (normalized).
- `bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}`: Lower and upper bounds for the battery capacity.
- `SoH_threshold::Float64`: State of health (SoH) level to replace the battery.
- `couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}`: Tuple to indicate if SoH should influence other parameters.

## Initial Conditions

- `Erated_ini::Float64`: Initial capacity of the battery in Wh.
- `soc_ini::Float64`: Initial state of charge for the beginning of the simulation.
- `soh_ini::Float64`: Initial state of health for the beginning of the simulation.

## Variables

- `Erated::AbstractArray{Float64, 2}`: Battery capacity.
- `carrier::Electricity`: Type of energy.
- `soc::AbstractArray{Float64, 3}`: 3-dimensional matrix (h, y, s) containing the state of charge [0-1].
- `soh::AbstractArray{Float64, 3}`: 3-dimensional matrix (h, y, s) containing the state of health [0-1].
## Eco
- `cost::AbstractArray{Float64, 2}`: Economic cost.

## Example 
```julia
Liion(eff_model = PolynomialLiionEfficiency(), SoH_model = FixedLifetimeLiion())```

"""
mutable struct Liion <: AbstractLiion

	
	eff_model::AbstractLiionEffModel
	SoH_model::AbstractLiionAgingModel

	# Parameters
	α_soc_min::Float64 #min threshold of charge (normalized)
	α_soc_max::Float64 #max threshold of charge (normalized)

	η_self::Float64 
	

	bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
	SoH_threshold::Float64 # SoH level to replace battery
	couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}  #a boolean tuple to tell wether or not the soh should influence the other parameters.


	# Initial conditions
	Erated_ini::Float64  # capacité de la batterie en Wh
	soc_ini::Float64 # first state of charge for the begining of simulation
	soh_ini::Float64 # first state of health for the begining of simulation

	# Variables
	Erated::AbstractArray{Float64,2} # Battery capacity
	carrier::Electricity #Type of energy
	soc::AbstractArray{Float64,3} #3 dim matrix (h,y,s) containing the state of charge [0-1]
	soh::AbstractArray{Float64,3} #3 dim matrix (h,y,s) containing the state of health [0-1]

	 # Eco
	cost::AbstractArray{Float64,2}

	# Inner constructor
	Liion(; eff_model = FixedLiionEfficiency(),
		SoH_model = FixedLifetimeLiion(),
		α_soc_min = 0.2,
		α_soc_max = 0.8,
		η_self = 0.0005,
		bounds = (lb = 0., ub = 1000.),
		SoH_threshold = 0.8,
		couplage = (E = true, R = true),
		Erated_ini = 1e-6,
		soc_ini = 0.5,
		soh_ini = 1.) = new(eff_model, SoH_model, α_soc_min, α_soc_max, η_self, bounds,
			SoH_threshold, couplage, Erated_ini, soc_ini, soh_ini) 

end

"""
# preallocate!

Preallocates arrays within the Liion struct for a given simulation size.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `nh::Int64`: Number of time steps.
- `ny::Int64`: Number of scenarios.
- `ns::Int64`: Number of states.

## Description

This function preallocates arrays within the Liion struct based on the simulation size defined by `nh`, `ny`, and `ns`. It initializes arrays such as `Erated`, `carrier.power`, `soc`, `soh`, and `cost` to improve efficiency during simulation.

## Example

```julia
liion = Liion()
preallocate!(liion, nh, ny, ns)
```
"""
function preallocate!(liion::Liion, nh::Int64, ny::Int64, ns::Int64)
   liion.Erated = convert(SharedArray,zeros(ny+1, ns)) ; liion.Erated[1,:] .= liion.Erated_ini
   liion.carrier = Electricity()
   liion.carrier.power = convert(SharedArray,zeros(nh, ny, ns))  
   liion.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soc[1,1,:] .= liion.soc_ini
   liion.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soh[1,1,:] .= liion.soh_ini
   liion.cost = convert(SharedArray,zeros(ny, ns))
  
   if liion.SoH_model isa EnergyThroughputLiion
 		liion.SoH_model.nCycle = Int(round(liion.SoH_model.nCycle_ini * 1/(1-liion.SoH_threshold))) #cycle to failure calibrée pour SoH_threshold %
   elseif  liion.SoH_model isa SemiEmpiricalLiion
		liion.SoH_model.Sum_fd = convert(SharedArray,zeros(ns))
   end

   liion.eff_model.couplage = liion.couplage

   return liion
end


"""
# compute_operation_dynamics!

Compute and update inner arrays of Liion according to the input decisions using [`compute_operation_soc`](@ref) and [`compute_operation_soh`](@ref).

## Arguments

- `h::Int64`: Operation time step index.
- `y::Int64`: Decision step index.
- `s::Int64`: scenario index.
- `liion::Liion`: Li-ion battery structure.
- `decision::Float64`: Input power decision (positive or negative).
- `Δh::Int64`: Time step duration.

## Description

This function computes and updates the state of charge (SoC), power, and state of health (SoH) of the Liion battery model based on the input decisions. It utilizes the [`compute_operation_soc`](@ref) and [`compute_operation_soh`](@ref) functions to perform the calculations.

## Example

```julia
h = 1
y = 1
s = 1
decision = 0.5
Δh = 1
compute_operation_dynamics!(h, y, s, liion, decision, Δh)```
"""
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, liion::Liion, decision::Float64, Δh::Int64)

	liion.soc[h+1,y,s], power_ch, power_dch = compute_operation_soc(liion, liion.eff_model, h ,y ,s , decision, Δh)
	
	liion.carrier.power[h,y,s] = power_ch + power_dch 

	if liion.Erated[y,s] == 0
		liion.soh[h+1,y,s] = 0
	else
		liion.soh[h+1,y,s] = compute_operation_soh(liion,  liion.SoH_model, h ,y ,s, Δh)
	end

end


"""
# compute_operation_soc

Compute and update the state of charge (SoC) dynamics based on the input decisions using the FixedLiionEfficiency model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::FixedLiionEfficiency`: Linear efficiency model.
- `h::Int64`: Operation time step index.
- `y::Int64`: Decision time step index.
- `s::Int64`: Scenario index.
- `decision::Float64`: Power input (positive or negative).
- `Δh::Int64`: Time step duration.

## Description

This function computes and updates the state of charge (SoC) dynamics of the Liion battery model based on the input decisions and the FixedLiionEfficiency model. It considers efficiency, capacity coupling, and efficiency coupling according to the model parameters.

## Example

```julia
h = 1
y = 1
s = 1
decision = 0.5
Δh = 1
compute_operation_soc(liion, model, h, y, s, decision, Δh)
```
"""
function compute_operation_soc(liion::Liion, model::FixedLiionEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
	
	if liion.Erated[y,s] == 0
		return 0,0,0
	end 
	
	if decision >= 0 
		η_ini = model.η_dch 
	else
		η_ini = model.η_ch    
	end

	if model.couplage.E
	    Erated = liion.Erated[y,s] * liion.soh[h,y,s]
	else
	    Erated = liion.Erated[y,s]
	end

	if model.couplage.R
		η = η_ini - (model.η_deg_coef * (1-liion.soh[h,y,s]))   #redondo iglesias 
	else
		η = η_ini
	end

	@inbounds power_dch = max(min(decision, model.α_p_dch * Erated, liion.soh[h,y,s] * liion.Erated[y,s] / Δh, η * (liion.soc[h,y,s] - liion.α_soc_min) * Erated / Δh), 0.)
	@inbounds power_ch = min(max(decision, -model.α_p_ch * Erated, -liion.soh[h,y,s] * liion.Erated[y,s] / Δh, (liion.soc[h,y,s] - liion.α_soc_max) * Erated / Δh / η), 0.)

	if Erated == 0
		return 0
	else
		return (1-model.η_self) * liion.soc[h,y,s] - (power_ch * η + power_dch / η) * Δh / Erated, power_ch, power_dch
	end

end

"""
# compute_operation_soc

Compute and update the state of charge (SoC) dynamics based on the input decisions using the PolynomialLiionEfficiency model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::PolynomialLiionEfficiency`: Polynomial efficiency model.
- `h::Int64`: Operation time step index.
- `y::Int64`: Decision time step index.
- `s::Int64`: Scenario index.
- `decision::Float64`: Power input (positive or negative).
- `Δh::Int64`: Time step duration.

## Description

This function computes and updates the state of charge (SoC) dynamics of the Liion battery model based on the input decisions and the PolynomialLiionEfficiency model. It considers efficiency, capacity coupling, and efficiency coupling according to the model parameters.

## Example

```julia
h = 1
y = 1
s = 1
decision = 0.5
Δh = 1
compute_operation_soc(liion, model, h, y, s, decision, Δh)
```
"""
function compute_operation_soc(liion::Liion, model::PolynomialLiionEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	
	if liion.Erated[y,s] == 0
		return 0,0,0
	end 
	
	if liion.couplage.E
		Erated = liion.Erated[y,s] * liion.soh[h,y,s]
	else
		Erated = liion.Erated[y,s]
	end
	
	C_r =  abs(decision * Δh) / Erated 
	
	if decision >= 0 
		η_ini = model.a_η_ch * C_r ^ 2 - model.b_η_ch * C_r + model.c_η_ch 
	else
		η_ini = model.a_η_dch * C_r ^ 2 - model.b_η_dch * C_r + model.c_η_dch 
	end

	if liion.couplage.R
		η = η_ini - (model.η_deg_coef  * (1-liion.soh[h,y,s])) 
	else
		η = η_ini
	end

	power_dch = max(min(decision, model.α_p_dch * Erated, liion.soh[h,y,s] * liion.Erated[y,s] / Δh, η * (liion.soc[h,y,s] - liion.α_soc_min) * Erated / Δh), 0.)
	power_ch = min(max(decision, -model.α_p_ch * Erated, -liion.soh[h,y,s] * liion.Erated[y,s] / Δh, (liion.soc[h,y,s] - liion.α_soc_max) * Erated / Δh / η), 0.)


	return (1-model.η_self) * liion.soc[h,y,s] - (power_ch * η + power_dch / η) * Δh / Erated, power_ch, power_dch
	

end


"""
# compute_operation_soh

Compute and update the state of health (SoH) dynamics at the current time using the EnergyThroughputLiion aging model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::EnergyThroughputLiion`: EnergyThroughputLiion aging model.
- `h::Int64`: Operation time step index.
- `y::Int64`: Decision time step index.
- `s::Int64`: Scenario index.
- `Δh::Int64`: Time step duration.

## Description

This function computes and updates the state of health (SoH) dynamics of the Liion battery model. It considers both cycling aging and calendar aging.
The EnergyThroughput model compute aging based on the energy going through the battery. It consider a total amount of exchangeable energy and compute the SoH based on a fraction of it already exchanged.
## Example

```julia
h = 1
y = 1
s = 1
Δh = 1
compute_operation_soh(liion, model, h, y, s, Δh)```
"""
function compute_operation_soh(liion::Liion, model::EnergyThroughputLiion, h::Int64,y::Int64 ,s::Int64 , Δh::Int64)
	
	ΔSoH = (abs(liion.carrier.power[h,y,s])) * Δh / (2. * model.nCycle * (liion.α_soc_max - liion.α_soc_min) * liion.Erated[y,s])


   #Calendar part
   if model.calendar == true
		ΔSoH += model.Δcal * Δh
   end

   return liion.soh[h,y,s] - ΔSoH

end


"""
# compute_operation_soh

Compute and update the state of health (SoH) dynamics based on the SemiEmpiricalLiion aging model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::SemiEmpiricalLiion`: SemiEmpiricalLiion aging model.
- `h::Int64`: Operation time step index.
- `y::Int64`: Decision time step index.
- `s::Int64`: Scenario index.
- `Δh::Int64`: Time step duration.

## Description

This function computes and updates the state of health (SoH) dynamics of the Liion battery model based on the SemiEmpiricalLiion aging model.  The SoH is updated based on rainflow cycle counting, considering calendar aging and cycling aging. It uses the `compute_operation_soh_rainflow` function with the state of charge (SoC) profile `liion.soc[interval, y, s]` over a certain interval.

## References
	
- [`compute_operation_soh_rainflow`](#compute_operation_soh_rainflow): Function for rainflow cycle counting. (implémented for this model and different form the one for the rainflow aging model)
- `liion.soc[interval, y, s]`: State of charge (SoC) profile over a certain interval.
	
## Example

```julia
h = 1
y = 1
s = 1
Δh = 1
compute_operation_soh(liion, model, h, y, s, Δh)```
"""
function compute_operation_soh(liion::Liion, model::SemiEmpiricalLiion, h::Int64, y::Int64, s::Int64, Δh::Int64)
	
	h_between_update = convert(Int64,floor(8760/model.update_by_year))
	#SoH computation
	if (h%h_between_update) != 0
		next_soh = liion.soh[h,y,s]
	else #rainflow computaion
		interval = (h-h_between_update+1):h

		next_soh, liion.SoH_model.Sum_fd[s] = compute_operation_soh_rainflow(liion, liion.SoH_model, Δh,  liion.soc[interval,y,s], liion.SoH_model.Sum_fd[s])
	end

	return next_soh
	
end

"""
# compute_operation_soh

Compute and update the state of health (SoH) dynamics based on the RainflowLiion aging model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::RainflowLiion`: RainflowLiion aging model.
- `h::Int64`: Operation time step index.
- `y::Int64`: Decision time step index.
- `s::Int64`: Scenario index.
- `Δh::Int64`: Time step duration.

## Description

This function computes and updates the state of health (SoH) dynamics of the Liion battery model based on the RainflowLiion aging model. The SoH is updated based on rainflow cycle counting, considering calendar aging and cycling aging. It uses the `compute_operation_soh_rainflow` function with the state of charge (SoC) profile `liion.soc[interval, y, s]` over a certain interval.

## References

- [`compute_operation_soh_rainflow`](#compute_operation_soh_rainflow): Function for rainflow cycle counting.
- `liion.soc[interval, y, s]`: State of charge (SoC) profile over a certain interval.

## Example

```julia
h = 1
y = 1
s = 1
Δh = 1
compute_operation_soh(liion, model, h, y, s, Δh)
```
"""
function compute_operation_soh(liion::Liion, model::RainflowLiion, h::Int64, y::Int64, s::Int64, Δh::Int64)
	
	h_between_update = convert(Int64,floor(8760/model.update_by_year))
	#SoH computation
	if (h%h_between_update) != 0
		next_soh = liion.soh[h,y,s]
	else #rainflow computaion
		interval = (h-h_between_update+1):h

		ΔSoH = compute_operation_soh_rainflow(liion, model, Δh,  liion.soc[interval,y,s])

		#Calendar part
		if model.calendar == true
            ΔSoH +=  h_between_update * (1 - exp(- 4.14e-10 * 3600 ))
        end

		next_soh = liion.soh[h,y,s] - ΔSoH
	end

	return next_soh
	
end

"""
# compute_operation_soh

Compute and update the state of health (SoH) dynamics based on the FixedLifetimeLiion aging model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::FixedLifetimeLiion`: FixedLifetimeLiion aging model.
- `h::Int64`: Operation time step index.
- `y::Int64`: Decision time step index.
- `s::Int64`: Scenario index.
- `Δh::Int64`: Time step duration.

## Description

This function computes and updates the state of health (SoH) dynamics of the Liion battery model based on the FixedLifetimeLiion aging model. The SoH is updated based on a fixed lifetime, and the update is proportional to the remaining lifetime of the battery.

## Example

```julia
h = 1
y = 1
s = 1
Δh = 1
compute_operation_soh(liion, model, h, y, s, Δh)
```
"""
function compute_operation_soh(liion::Liion, model::FixedLifetimeLiion, h::Int64, y::Int64, s::Int64, Δh::Int64)
	
	return liion.soh[h,y,s] - ((1 - liion.SoH_threshold) * Δh)/(8760 * model.lifetime)
	
end





"""
# compute_operation_soh_rainflow

Compute and update the state of health (SoH) using rainflow cycle counting based on the SemiEmpiricalLiion aging model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::SemiEmpiricalLiion`: SemiEmpiricalLiion aging model.
- `Δh::Int64`: Time step duration.
- `soc::Vector{Float64}`: State of charge (SoC) profile over a certain interval.
- `Sum_fd::Float64`: Cumulated fatigue of the battery.

## Description

This function calculates the state of health (SoH) of the Liion battery model using rainflow cycle counting based on the SemiEmpiricalLiion aging model. It analyzes the SoC profile, identifies peaks, and computes the Depth of Discharge (DoD) sequences for each sub-cycle. The cumulated fatigue is updated, and the overall SoH is determined.

## Example

```julia
Δh = 1
soc_profile = [0.2, 0.4, 0.8, 0.6, 0.2]
Sum_fd = 0.0
compute_operation_soh_rainflow(liion, model, Δh, soc_profile, Sum_fd)
```
"""
function compute_operation_soh_rainflow(liion::Liion, model::SemiEmpiricalLiion, Δh::Int64, soc::Vector{Float64}, Sum_fd::Float64)

	soc_peak, soc_peak_id = get_soc_peaks(soc)

	#Then compute the DoD sequence by extracting the subcycles DoD

	DoD_seq = Float64[] #Sequence of all the charging and decharging half cycles DoDs
	mean_Soc_seq = Float64[]
	delta_t_seq = Int64[]

	i = 1


	while i+3 <= length(soc_peak_id)	#Define your 3 deltas with 4 consecutives points
		delta1 = abs( soc[soc_peak_id[i+1]] - soc[soc_peak_id[i]] )
	 	delta2 = abs( soc[soc_peak_id[i+2]] - soc[soc_peak_id[i+1]] )
	 	delta3 = abs( soc[soc_peak_id[i+3]] - soc[soc_peak_id[i+2]] )

	 	#rainflow sub-cycle criterion
	 	if delta2 <= delta1 && delta2 <= delta3
			push!(DoD_seq, delta2) #1 half cycle of DoD delta2 +
			push!(DoD_seq, delta2) #1 half cycle
			push!(mean_Soc_seq, (soc[soc_peak_id[i+2]] + soc[soc_peak_id[i+1]]) /2 ) #SoC mean
			push!(mean_Soc_seq, (soc[soc_peak_id[i+2]] + soc[soc_peak_id[i+1]]) /2 ) #SoC mean
			push!(delta_t_seq, soc_peak_id[i+2] - soc_peak_id[i+1])#delta_t = soc_peak_id[i+2] - soc_peak_id[i+1] but we dont want to count the time twice
			push!(delta_t_seq, soc_peak_id[i+2] - soc_peak_id[i+1])#delta_t, we use the time of the second half cycle because the first one is altered by the algorithme.


	 		deleteat!(soc_peak_id, i+2) #start with i+2 index or you will delete i+1 and i+3
	 		deleteat!(soc_peak_id, i+1)
		else #else use the following point sequence
	 		i = i+1
	 	end
 	end

 	#Then add the englobing (those who make the other cycles "sub") cycles to the DoD sequence*

	for i in 1:(length(soc_peak_id)-1)
		push!(DoD_seq, abs( soc[soc_peak_id[i+1]] - soc[soc_peak_id[i]] )) #DoD
		push!(mean_Soc_seq, (soc[soc_peak_id[i+1]] + soc[soc_peak_id[i]]) /2 ) #SoC mean
		if length(soc_peak_id) <= 3 #If there is no inner cycles we need to count the time here
			push!(delta_t_seq, soc_peak_id[i+1] - soc_peak_id[i])
		else
			push!(delta_t_seq, 0)
		end
	end


	for i in 1:length(DoD_seq)
		Sum_fd += compute_fd(model,  DoD_seq[i] , model.temperature, mean_Soc_seq[i], delta_t_seq[i])
	end


	L = 1 - ( model.alpha_sei * exp(-model.beta_sei * Sum_fd) ) - ( (1 - model.alpha_sei) * exp(-Sum_fd ))

	#SOH = 1- L
	return 1 - L, Sum_fd

end


"""
# compute_operation_soh_rainflow

Compute and update the state of health (SoH) using rainflow cycle counting based on the RainflowLiion aging model.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `model::RainflowLiion`: RainflowLiion aging model.
- `Δh::Int64`: Time step duration.
- `soc::Vector{Float64}`: State of charge (SoC) profile.

## Description

This function calculates the state of health (SoH) of the Liion battery model using rainflow cycle counting based on the RainflowLiion aging model. It identifies peaks in the SoC profile and computes the Depth of Discharge (DoD) sequences for each sub-cycle. The cumulated fatigue is determined, and the overall SoH is updated.

## Example

```julia
Δh = 1
soc_profile = [0.2, 0.4, 0.8, 0.6, 0.2]
compute_operation_soh_rainflow(liion, model, Δh, soc_profile)
```
"""
function compute_operation_soh_rainflow(liion::Liion, model::RainflowLiion, Δh::Int64, soc::Vector{Float64})

	#Gather peaks from the soc profil
	soc_peak, _ = get_soc_peaks(soc)


	#Then compute the DoD sequence by extracting the subcycles DoD
	DoD_seq = Float64[] #Sequence of all the charging and decharging half cycles DoDs

	i = 1

	while i+3 <= length(soc_peak)
		#Define your 3 deltas with 4 consecutives points
		delta1 = abs( soc_peak[i+1] - soc_peak[i] )
		delta2 = abs( soc_peak[i+2] - soc_peak[i+1] )
		delta3 = abs( soc_peak[i+3] - soc_peak[i+2] )

		#rainflow sub-cycle criterion
		if delta2 <= delta1 && delta2 <= delta3
			push!(DoD_seq, delta2) #1 half cycle of DoD delta2 +
			push!(DoD_seq, delta2) #1 half cycle
			deleteat!(soc_peak, i+2) #start with the second or you will delete i+1 and i+3
			deleteat!(soc_peak, i+1)
		else #else use the following point sequence
			i = i+1
		end
	end

	#Then add the englobing (those who make the other cycles "sub") cycles to the DoD sequence*
	for i in 1:(length(soc_peak)-1)
		push!(DoD_seq, abs(soc_peak[i+1]-soc_peak[i]))
	end

	#currently neglect cycle under 1%
	#deleteat!(DoD_seq, findall(<(1e-2), DoD_seq))

	if length(DoD_seq) > 0 && isnan(DoD_seq[1])
		println("soc : ", soc)
	end
	fatigue = 0

	for i in 1:length(DoD_seq)
		fatigue += 1/(2*Φ(DoD_seq[i], model.fatigue_data) ) #Compute fatigue with phy function applied to all the half cycles DoD factor 2 refer to half cycles
	end

	return (fatigue * (1 - liion.SoH_threshold)) #/5 car la courbe cycle to failure donne le nombre de cycle jusqu'à 80% SOH
end



"""
# compute_operation_dynamics

Compute and retur the power dynamics based on the input decisions.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `h::Int64`: Operation time step index.
- `y::Int64`: Decision time step index.
- `s::Int64`: Scenario index.
- `decision::Float64`: Power input (positive or negative).
- `Δh::Int64`: Time step duration.

## Description

This function computes the power dynamics of the Liion battery model based on the input decisions. It calls the `compute_operation_soc` function to calculate the next state of charge (SoC), power for charging (`power_ch`), and power for discharging (`power_dch`). 

## Example

```julia
h = 1
y = 1
s = 1
decision = -10.0
Δh = 1
compute_operation_dynamics(liion, h, y, s, decision, Δh)
```
"""
function compute_operation_dynamics(liion::Liion, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)
	 
	soc_next, power_ch, power_dch = compute_operation_soc(liion, liion.eff_model, h, y, s, decision, Δh)
	
	return power_dch + power_ch, soc_next
 end
 
 """
 # compute_investment_dynamics!
 
 Compute and update Liion battery model arrays for investment dynamics based on the input decision.
 
 ## Arguments
 
 - `y::Int64`: Decision time step index.
 - `s::Int64`: Scenario index.
 - `liion::Liion`: Li-ion battery model.
 - `decision::Union{Float64, Int64}`: Investment decision.
 
 ## Description
 
 This function computes and updates the Liion battery model arrays for investment dynamics based on the input decision. It calls the `compute_investment_dynamics` function to calculate the new values for battery capacity (`Erated`), initial state of charge (`soc`), and initial state of health (`soh`). The Liion battery model arrays are updated in place.
 
 ## Example
 
 ```julia
 y = 1
 s = 1
 decision = 100.0
 compute_investment_dynamics!(y, s, liion, decision)
```
"""
 function compute_investment_dynamics!(y::Int64, s::Int64, liion::Liion, decision::Union{Float64, Int64})
	liion.Erated[y+1,s], liion.soc[1,y+1,s], liion.soh[1,y+1,s] = compute_investment_dynamics(liion, (Erated = liion.Erated[y,s], soc = liion.soc[end,y,s], soh = liion.soh[end,y,s]), decision, s)
 end


 """
# initialize_investments!

Initialize (at first year) Liion battery model arrays for investments.

## Arguments

- `s::Int64`: Scenario index.
- `liion::Liion`: Li-ion battery model.
- `decision::Union{Float64, Int64}`: Initial investment decision.

## Description

This function initializes the Liion battery model arrays for investments based on the initial investment decision. It sets the initial values for battery capacity (`Erated`), initial state of charge (`soc`), and initial state of health (`soh`). The Liion battery model arrays are updated in place.

## Example

```julia
s = 1
decision = 100.0
initialize_investments!(s, liion, decision)
```
"""
 function initialize_investments!(s::Int64, liion::Liion, decision::Union{Float64, Int64})
   liion.Erated[1,s] = decision
   liion.soc[1,1,s] = liion.soc_ini
   liion.soh[1,1,s] = liion.soh_ini

   if liion.SoH_model isa SemiEmpiricalLiion
	liion.SoH_model.Sum_fd[s] = 0.
   end
end


"""
# compute_investment_dynamics

Compute Liion battery model dynamics for investment decisions.

## Arguments

- `liion::Liion`: Li-ion battery model.
- `state::NamedTuple{(:Erated, :soc, :soh), Tuple{Float64, Float64, Float64}}`: Current state of the battery model.
- `decision::Union{Float64, Int64}`: Investment decision.

## Returns

- `(Erated_next, soc_next, soh_next)`: Updated values for battery capacity (`Erated`), state of charge (`soc`), and state of health (`soh`).

## Description

This function computes the Liion battery model dynamics for investment decisions. It calculates the updated values for battery capacity, state of charge, and state of health based on the given investment decision. If the investment decision is greater than `1e-2`, it sets the battery to a new state, otherwise, it maintains the current state.

The Liion battery model arrays are updated in place.

## Example

```julia
state = (Erated = 100.0, soc = 0.5, soh = 1.0)
decision = 50.0
Erated_next, soc_next, soh_next = compute_investment_dynamics(liion, state, decision)
```
"""
 function compute_investment_dynamics(liion::Liion, state::NamedTuple{(:Erated, :soc, :soh), Tuple{Float64, Float64, Float64}}, decision::Union{Float64, Int64}, s::Int64)
	 if decision > 1e-2
		 Erated_next = decision
		 soc_next = liion.soc_ini
		 soh_next =  liion.soh_ini
		 if liion.SoH_model isa SemiEmpiricalLiion
			liion.SoH_model.Sum_fd[s] = 0.
		 end
	 else
		 Erated_next = state.Erated
		 soc_next = state.soc
		 soh_next = state.soh
	 end


	 return Erated_next, soc_next, soh_next
 end




 """
 # get_soc_peaks
 
 Identify peaks in the state of charge (soc) vector.
 
 ## Arguments
 
 - `soc::Vector{Float64}`: Vector containing the state of charge values.
 
 ## Returns
 
 - `(soc_peak, soc_peak_id)`: Tuple containing sequences of state of charge peaks (`soc_peak`) and their corresponding indices (`soc_peak_id`).
 
 ## Description
 
 This function identifies peaks in the given state of charge vector (`soc`). It returns two sequences: `soc_peak`, which contains the values of the state of charge peaks, and `soc_peak_id`, which contains the indices of these peaks in the original vector.
 
 The identification is based on changes in the sign of the first derivative of the state of charge vector. A peak is detected when there is a change in trend (from increasing to decreasing or vice versa).
 
 ## Example
 
 ```julia
 soc = [0.2, 0.5, 0.8, 0.4, 0.9, 0.3, 0.7]
 soc_peak, soc_peak_id = get_soc_peaks(soc)
```
""" 
function get_soc_peaks(soc::Vector{Float64})
	soc_peak = Float64[] #soc_peak is the sequence of values for the state of charges peaks
	soc_peak_id = Int64[] #soc_peak is the sequence of values for the state of charges peaks

	#add first
	push!(soc_peak, soc[1])
	push!(soc_peak_id, 1)

	# Extract all peak value of soc
	for i in 2:(length(soc)-1)
		sign1 = soc[i] - soc[i-1] > 0 #true = positiv, false = negativ
		sign2 = soc[i+1] - soc[i] > 0

		if sign1 != sign2 # different sign mean change of trend (increasing of decreasing) so it's a peak
			push!(soc_peak, soc[i])
			push!(soc_peak_id, i)
		end
	end



	#add last
	push!(soc_peak, soc[length(soc)])
	push!(soc_peak_id, length(soc))

	return soc_peak, soc_peak_id
end


"""
# S_delta

Compute a factor based on depth of discharge (DoD) for a SemiEmpiricalLiion model.

## Arguments

- `params::SemiEmpiricalLiion`: Parameters of the SemiEmpiricalLiion model.
- `DoD::Float64`: Depth of discharge.

## Returns

- `Float64`: Depth of discharge factor.

## Description

This function calculates a factor (`S_delta`) based on the depth of discharge (DoD) for a SemiEmpiricalLiion model. The factor is computed using the parameters specified in the `params` argument.

## Example

```julia
DoD_factor = S_delta(params, 0.5)
```
"""
function S_delta(params::SemiEmpiricalLiion, DoD::Float64)
	return params.k_delta1 * (DoD ^ params.k_delta2)
end

"""
# S_T

Compute a temperature-related factor for a SemiEmpiricalLiion model.

## Arguments

- `params::SemiEmpiricalLiion`: Parameters of the SemiEmpiricalLiion model.
- `T::Float64`: Temperature.

## Returns

- `Float64`: Temperature factor.

## Description

This function calculates a factor (`S_T`) based on temperature for a SemiEmpiricalLiion model. The factor is computed using the parameters specified in the `params` argument.

## Example

```julia
temperature_factor = S_T(params, 3
```
"""
function S_T(params::SemiEmpiricalLiion, T::Float64)
	return exp(params.k_T*(T-params.T_ref) * (params.T_ref/T))
end

"""
# S_sigma

Compute a factor related to mean state of charge during cycles for a SemiEmpiricalLiion model.

## Arguments

- `params::SemiEmpiricalLiion`: Parameters of the SemiEmpiricalLiion model.
- `mean_SoC::Float64`: Mean state of charge.

## Returns

- `Float64`: State of charge factor.

## Description

This function calculates a factor (`S_sigma`) based on the mean state of charge for a SemiEmpiricalLiion model. The factor is computed using the parameters specified in the `params` argument.

## Example

```julia
soc_factor = S_sigma(params, 0.6)
```
"""
function S_sigma(params::SemiEmpiricalLiion, mean_SoC::Float64)
	return exp(params.k_sigma * (mean_SoC  - params.sigma_ref))
end


"""
# S_t

Compute a time-related factor for a SemiEmpiricalLiion model.

## Arguments

- `params::SemiEmpiricalLiion`: Parameters of the SemiEmpiricalLiion model.
- `t::Int64`: Time in hours.

## Returns

- `Float64`: Time factor.

## Description

This function calculates a factor (`S_t`) based on the time for a SemiEmpiricalLiion model. The factor is computed using the parameters specified in the `params` argument.

## Example

```julia
time_factor = S_t(params, 10)
```
"""
function S_t(params::SemiEmpiricalLiion, t::Int64)
	return params.k_t * t * 3600 #hours to second
end


"""
# compute_fd

Compute the Fatigue Damage for a SemiEmpiricalLiion model.

## Arguments

- `params::SemiEmpiricalLiion`: Parameters of the SemiEmpiricalLiion model.
- `DoD::Float64`: Depth of Discharge.
- `T::Float64`: Temperature.
- `mean_SoC::Float64`: Mean State of Charge.
- `t::Int64`: Time in hours.

## Returns

- `Float64`: Fatigue Damage.

## Description

This function calculates the Fatigue Damage (`compute_fd`) for a SemiEmpiricalLiion model. The damage is computed based on the Depth of Discharge (`DoD`), Temperature (`T`), Mean State of Charge (`mean_SoC`), and time (`t`) using the parameters specified in the `params` argument.

## Example

```julia
damage = compute_fd(params, 0.2, 298, 0.5, 1)
```
"""
function compute_fd(params::SemiEmpiricalLiion, DoD::Float64, T::Float64, mean_SoC::Float64, t::Int64)

	return (0.5 * S_delta(params, DoD) + S_t(params, t)) * S_sigma(params, mean_SoC) * S_T(params, T)
end


"""
# Φ

Compute the fatigue cycle corresponding to a given Depth of Discharge (DoD).

## Arguments

- `DoD::Float64`: Depth of Discharge.
- `fatigue_data`: Fatigue data containing the mapping between DoD and fatigue cycles.

## Returns

- `Any`: Fatigue cycle corresponding to the given DoD.

## Description

This function calculates the fatigue cycle (`Φ`) based on the provided Depth of Discharge (`DoD`) and the fatigue data. The `fatigue_data` is expected to have a field `DoD` containing the DoD values and a field `cycle` containing the corresponding fatigue cycles.

## Example

```julia
data = (DoD = [0.1, 0.2, 0.3], cycle = [1000, 800, 600])
cycle = Φ(0.25, data)
```
"""
function Φ(DoD::Float64, fatigue_data)
	index = findfirst(>=(DoD), fatigue_data.DoD)

	return fatigue_data.cycle[index]
end


function toStringShort(liion::Liion)

	if liion.eff_model isa FixedLiionEfficiency
		efficiency = "x"
	elseif liion.eff_model isa PolynomialLiionEfficiency
		efficiency = "x²"
	end

	if liion.SoH_model isa FixedLifetimeLiion
		aging = "FL"
	elseif liion.SoH_model isa EnergyThroughputLiion
		aging = "ET"
	elseif liion.SoH_model isa RainflowLiion
		aging = "RF"
	elseif liion.SoH_model isa SemiEmpiricalLiion
		aging = "SE"

	end

	return string("Liion :", efficiency, ", ", aging)
end