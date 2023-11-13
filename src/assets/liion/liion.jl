abstract type AbstractLiion <: AbstractStorage  end

abstract type AbstractLiionEffModel end

abstract type AbstractLiionAgingModel end

"""
LinearLiionEfficiency

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
LinearLiionEfficiency()
```
"""
mutable struct LinearLiionEfficiency <: AbstractLiionEffModel

	η_ch::Float64 #Charging efficiency
	η_dch::Float64 #Discharging efficiency
	η_deg_coef::Float64 #The efficiency degradation coefficient
	couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}  #a boolean tuple to tell wether or not the soh should influence the other parameters.
	α_p_ch::Float64 #C_rate max
	α_p_dch::Float64 #C_rate max
	η_self::Float64 #Auto discarge factor
	
	
	LinearLiionEfficiency(;η_ch = 0.98,
		η_dch = 0.98,
		η_deg_coef = 0.2303,  # ref : Redondo Iglesias - Efficiency Degradation Model of Lithium-Ion Batteries for Electric Vehicles
		couplage = (E = true, R = true),
		α_p_ch = 1.5,
		α_p_dch = 1.5,
		η_self = 0.0005,
		) = new(η_ch, η_dch, η_deg_coef, couplage, α_p_ch, α_p_dch ,η_self)

end

"""
polynomialLiionEfficiency

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
polynomialLiionEfficiency()
```
"""
mutable struct polynomialLiionEfficiency <: AbstractLiionEffModel

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
	
	
	polynomialLiionEfficiency(;a_η_ch=	0.0033,
	b_η_ch = 0.0297, 
	c_η_ch = 0.99814,
	a_η_dch = 0.002232, 
	b_η_dch = 0.0246, 
	c_η_dch = 1,
	η_deg_coef = 0.2303, # ref : Redondo Iglesias - Efficiency Degradation Model of Lithium-Ion Batteries for Electric Vehicles
	couplage = (E = true, R = false),
	α_p_ch = 1.5,
	α_p_dch = 1.5,
	η_self = 0.0005) = new(a_η_ch, b_η_ch, c_η_ch, a_η_dch, b_η_dch, c_η_dch, η_deg_coef, couplage, α_p_ch, α_p_dch ,η_self)

end


"""
EnergyThroughputLiion

A mutable struct that represents an aging model for Li-ion batteries based on energy throughput, which accounts for calendar aging and cycling aging.
This model calculates aging based on the cumulative energy throughput and additionnaly calendar aging.

# Parameters:
- `calendar::Bool`: A boolean value to indicate whether to consider calendar aging (default: true)
- `nCycle::Int64`: The total number of cycles (default: 2500)
- `Δcal::Float64`: The calendar aging parameter (default: 1 - exp(-4.14e-10 * 3600))

## Example 
```julia
EnergyThroughputLiion()
```
"""
mutable struct EnergyThroughputLiion <: AbstractLiionAgingModel

	calendar::Bool
	nCycle::Int64
	Δcal::Float64 

	EnergyThroughputLiion(;calendar = true,
	nCycle = 2500.,
	Δcal = (1 - exp(- 4.14e-10 * 3600))
	) = new(calendar, nCycle, Δcal)
end

"""
FixedLifetimeLiion

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
RainflowLiion

A mutable struct that represents an aging model for Li-ion batteries based on rainflow cycle counting, considering calendar aging and cycling aging.

# Parameters:
- `calendar::Bool`: A boolean value for the activation of calendar aging (default: true)
- `update_by_year::Int64`: Number of updates each year (default: 12)
- `fatigue_data::DataFrames.DataFrame`: Cycle to failure curve data (default: loaded from a CSV file with a DoD column and a nCycle column)

## Example 
```julia
RainflowLiion()
```
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
SemiEmpiricalLiion

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
- `temperature::Float64`: Temperature of the battery (default: 298K)
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




mutable struct Liion <: AbstractLiion

	
	SoC_model::AbstractLiionEffModel
	SoH_model::AbstractLiionAgingModel

	# Parameters
	α_soc_min::Float64 #min threshold of charge (normalized)
	α_soc_max::Float64 #max threshold of charge (normalized)

	

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
	Liion(; SoC_model = LinearLiionEfficiency(),
		SoH_model = EnergyThroughputLiion(),
		α_soc_min = 0.2,
		α_soc_max = 0.8,
		bounds = (lb = 0., ub = 1000.),
		SoH_threshold = 0.8,
		couplage = (E = true, R = true),
		Erated_ini = 1e-6,
		soc_ini = 0.5,
		soh_ini = 1.) = new(SoC_model, SoH_model, α_soc_min, α_soc_max, bounds,
			SoH_threshold, couplage, Erated_ini, soc_ini, soh_ini) 

end

### Preallocation
function preallocate!(liion::Liion, nh::Int64, ny::Int64, ns::Int64)
   liion.Erated = convert(SharedArray,zeros(ny+1, ns)) ; liion.Erated[1,:] .= liion.Erated_ini
   liion.carrier = Electricity()
   liion.carrier.power = convert(SharedArray,zeros(nh, ny, ns))  
   liion.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soc[1,1,:] .= liion.soc_ini
   liion.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soh[1,1,:] .= liion.soh_ini
   liion.cost = convert(SharedArray,zeros(ny, ns))
  
   if liion.SoH_model isa EnergyThroughputLiion
 		liion.SoH_model.nCycle = Int(round(liion.SoH_model.nCycle * 1/(1-liion.SoH_threshold))) #cycle to failure calibrée pour SoH_threshold %
   elseif  liion.SoH_model isa SemiEmpiricalLiion
		liion.SoH_model.Sum_fd = convert(SharedArray,zeros(ns))
   end

   liion.SoC_model.couplage = liion.couplage

   return liion
end


### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, liion::Liion, decision::Float64, Δh::Int64)


	liion.soc[h+1,y,s], power_ch, power_dch = compute_operation_soc(liion, liion.SoC_model, h ,y ,s , decision, Δh)
	
	liion.carrier.power[h,y,s] = power_ch + power_dch 

	liion.soh[h+1,y,s] = compute_operation_soh(liion,  liion.SoH_model, h ,y ,s, Δh)


end


function compute_operation_soc(liion::Liion, model::LinearLiionEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	if decision >= 0 
		η_ini = model.η_dch #0.98   
	else
		η_ini = model.η_ch #0.98   
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

	power_dch = max(min(decision, model.α_p_dch * Erated, liion.soh[h,y,s] * liion.Erated[y,s] / Δh, η * (liion.soc[h,y,s] - liion.α_soc_min) * Erated / Δh), 0.)
	power_ch = min(max(decision, -model.α_p_ch * Erated, -liion.soh[h,y,s] * liion.Erated[y,s] / Δh, (liion.soc[h,y,s] - liion.α_soc_max) * Erated / Δh / η), 0.)

	return (1-model.η_self) * liion.soc[h,y,s] - (power_ch * η + power_dch / η) * Δh / Erated, power_ch, power_dch

end


function compute_operation_soc(liion::Liion, model::polynomialLiionEfficiency, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Int64)
	if liion.couplage.E
		Erated = liion.Erated[y,s] * liion.soh[h,y,s]
	else
		Erated = liion.Erated[y,s]
	end
	
	C_r = abs(decision * Δh) / Erated 
	
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



function compute_operation_soh(liion::Liion, model::EnergyThroughputLiion, h::Int64,y::Int64 ,s::Int64 , Δh::Int64)
	
	ΔSoH = (abs(liion.carrier.power[h,y,s])) * Δh / (2. * model.nCycle * (liion.α_soc_max - liion.α_soc_min) * liion.Erated[y,s])


   #Calendar part
   if model.calendar == true
		ΔSoH += model.Δcal * Δh
   end

   return liion.soh[h,y,s] - ΔSoH

end


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


function compute_operation_soh(liion::Liion, model::FixedLifetimeLiion, h::Int64, y::Int64, s::Int64, Δh::Int64)
	
	return liion.soh[h,y,s] - ((1 - liion.SoH_threshold) * Δh)/(8760 * model.lifetime)
	
end






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
			push!(delta_t_seq, soc_peak_id[i+2] - soc_peak_id[i+1])#delta_t
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
		push!(delta_t_seq, 0)#soc_peak_id[i+1] - soc_peak_id[i])
	end



	for i in 1:length(DoD_seq)
		Sum_fd += compute_fd(model,  DoD_seq[i] , model.temperature, mean_Soc_seq[i], delta_t_seq[i])
	end

	L = 1 - ( model.alpha_sei * exp(-model.beta_sei * Sum_fd) ) - ( (1 - model.alpha_sei) * exp(-Sum_fd ))

	#SOH = 1- L
	return 1 - L, Sum_fd

end



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


function compute_operation_dynamics(liion::Liion, h::Int64, y::Int64, s::Int64, decision::Float64, Δh::Int64)
	 
	soc_next, power_ch, power_dch = compute_operation_soc(liion, liion.SoC_model, h, y, s, decision, Δh)
	
	return power_dch + power_ch
 end
 
 ### Investment dynamic
 function compute_investment_dynamics!(y::Int64, s::Int64, liion::Liion, decision::Union{Float64, Int64})
	liion.Erated[y+1,s], liion.soc[1,y+1,s], liion.soh[1,y+1,s] = compute_investment_dynamics(liion, (Erated = liion.Erated[y,s], soc = liion.soc[end,y,s], soh = liion.soh[end,y,s]), decision)
 end


 
 function initialize_investments!(s::Int64, liion::Liion, decision::Union{Float64, Int64})
   liion.Erated[1,s] = decision
   liion.soc[1,1,s] = liion.soc_ini
   liion.soh[1,1,s] = liion.soh_ini
end

 function compute_investment_dynamics(liion::Liion, state::NamedTuple{(:Erated, :soc, :soh), Tuple{Float64, Float64, Float64}}, decision::Union{Float64, Int64})
	 if decision > 1e-2
		 Erated_next = decision
		 soc_next = liion.soc_ini
		 soh_next =  1.
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







 function get_power_flow(liion::AbstractLiion, state::NamedTuple{(:Erated, :soc, :soh), Tuple{Float64, Float64, Float64}}, decision::Float64, Δh::Int64)
	if liion.couplage.E
	 Erated = state.Erated * state.soh
	else
	 Erated = state.Erated
	end

   η_ini = 0.95   #Fixed (dis)charging efficiency for both BES and EV (0.98)     dans la nomenclature

   if liion.couplage.R
	   η = η_ini - ((1-state.soh)/12)   #(15) simplifié
   else
	   η = η_ini
   end

   	power_dch = max(min(decision, liion.α_p_dch * Erated, state.soh * state.Erated / Δh, η * (state.soc - liion.α_soc_min) * Erated / Δh), 0.)
	power_ch = min(max(decision, -liion.α_p_ch * Erated, -state.soh * state.Erated / Δh, (state.soc - liion.α_soc_max) * Erated / Δh / η), 0.)

   return power_dch, power_ch
end


#Optimal Sizing and Control of a PV-EV-BES Charging System Including Primary Frequency Control and Component Degradation
#Wiljan Vermeer et al.
#With this soc the efficiency is based on battery state of health
 function compute_operation_soc_Vermeer(liion::AbstractLiion, state::NamedTuple{(:Erated, :soc, :soh), Tuple{Float64, Float64, Float64}}, decision, Δh::Int64)

	η_ini = 0.98   #Fixed (dis)charging efficiency for both BES and EV (0.98)     dans la nomenclature
	η = η_ini - ((1-state.soh)/12)   #(15) simplifié

	power_dch = max(min(decision, liion.α_p_dch * state.Erated, state.soh *  state.Erated / Δh, η * (liion.α_soc_max - state.soc) * state.Erated * state.soh / Δh), 0.)
 	power_ch = min(max(decision, -liion.α_p_ch * state.Erated, -state.soh *  state.Erated / Δh, (liion.α_soc_min - state.soc) * state.Erated * state.soh / Δh / η), 0.)

	P = ( power_dch / η) + (power_ch * η) #(13)
	E_lim = state.Erated * state.soh   #Definition : Maximum battery capacity at time t, based on degradation
	E = E_lim * state.soc # based on (31)
	new_E = E + P * Δh #(34)

	return new_E/E_lim , P# Get the SoC to keep coherence with the entire code.
  end

function compute_operation_soc_artificial(liion::AbstractLiion, profil::Array{Float64,2},  y::Int64, h::Int64)
	return profil[y,h]
end



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



function S_delta(params::SemiEmpiricalLiion, DoD::Float64)
	return params.k_delta1 * (DoD ^ params.k_delta2)
end

function S_T(params::SemiEmpiricalLiion, T::Float64)
	return exp(params.k_T*(T-params.T_ref) * (params.T_ref/T))
end

function S_sigma(params::SemiEmpiricalLiion, mean_SoC::Float64)
	return exp(params.k_sigma * (mean_SoC  - params.sigma_ref))
end

function S_t(params::SemiEmpiricalLiion, t::Int64)
	return params.k_t * t * 3600 #hours to second
end

function compute_fd(params::SemiEmpiricalLiion, DoD::Float64, T::Float64, mean_SoC::Float64, t::Int64)
	return (0.5 * S_delta(params, DoD) + S_t(params, t)) * S_sigma(params, mean_SoC) * S_T(params, T)
end


function Φ(DoD::Float64, fatigue_data)
	index = findfirst(>=(DoD), fatigue_data.DoD)

	return fatigue_data.cycle[index]
end

