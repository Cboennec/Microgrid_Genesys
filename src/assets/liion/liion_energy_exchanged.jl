#TODO Le rendement utilisé ne doit pas être défini dans la fonction  compute_operation_soc_linear
#TODO Ici c'est fait dans compute_operation_dynamics
#=
    Li-ion battery modelling
 =#
 """
	Liion_energy_exchanged

A mutable struct that represents a Li-ion battery model Energy Throughput aging model for State of Health (SoH) computation.

The structure have a lot of parameters but most of them are set to default values.

# Parameters:
- `α_p_ch::Float64`: Charging maximum C-rate (default : 1.5)
- `α_p_dch::Float64`: Discharging  maximum C-rate (default : 1.5)
- `α_soc_min::Float64`: Minimum threshold of charge (normalized) (default : 0.2)
- `α_soc_max::Float64`: Maximum threshold of charge (normalized) (default : 0.8)
- `nCycle::Float64`: Number of cycle before reaching EOL (should be found in the cycle to failure curve)
- `SoH_threshold::Float64`: SoH level to replace the battery (default : 0.8)
- `couplage::NamedTuple`: Named tuple with two boolean values to indicate if the SoH should influence the other parameters (E stand for capacity coupling and R for efficiency coupling)
- `soc_model::String`: Model name for State of Charge (SoC) computation. Available models are listed 
- `calendar::Bool`: Whether to include calendar aging in the SoH computation  (default : true)
- `soc_ini::Float64`: Initial State of Charge (SoC) for the beginning of the simulation (default : 0.5)
- `soh_ini::Float64`: Initial State of Health (SoH) for the beginning of the simulation (default : 1)

## Example 
```julia
Liion_energy_exchanged(;calendar = true, nCycle = fatigue_data.cycle[findfirst(fatigue_data.DoD .> (0.6))], soc_model = "polynomial", couplage = (E=true, R=true))
```

Here the `nCycle` is selected from the cycle to failure curve using 60% DoD.
"""
 mutable struct Liion_energy_exchanged <: AbstractLiion


     # Parameters
 	α_p_ch::Float64
 	α_p_dch::Float64
 	η_ch::Float64 #Charging yield / efficacity
 	η_dch::Float64 #Discharging yield / efficacity
 	η_self::Float64 #Auto discarge factor
 	α_soc_min::Float64 #min threshold of charge (normalized)
 	α_soc_max::Float64 #max threshold of charge (normalized)
 	lifetime::Int64
 	nCycle::Float64
 	bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}}
 	SoH_threshold::Float64 # SoH level to replace battery
 	couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}  #a boolean tuple to tell wether or not the soh should influence the other parameters.

 	#Model dynamics
 	soc_model::String #model name
    calendar::Bool

 	# Initial conditions
 	Erated_ini::Float64  # capacité de la batterie en Wh
 	soc_ini::Float64 # first state of charge for the begining of simulation
 	soh_ini::Float64 # first state of health for the begining of simulation

    artificial_soc_profil::Array{Float64,2} # soc profil for study purpose


 	# Variables
 	Erated::AbstractArray{Float64,2} # Battery capacity
 	carrier::Electricity #Type of energy
 	soc::AbstractArray{Float64,3} #3 dim matrix (h,y,s) containing the state of charge [0-1]
 	soh::AbstractArray{Float64,3} #3 dim matrix (h,y,s) containing the state of health [0-1]
 	#tremblay_dessaint
 	voltage ::Array{Float64,3} # evolution de la tension
 	current ::Array{Float64,3} # evolution du courant ?
 	tremblay_dessaint_params::Tremblay_dessaint_params # A list of params only use for tremblay dessaint

  	# Eco
 	cost::AbstractArray{Float64,2}

 	# Inner constructor
 	Liion_energy_exchanged(; α_p_ch = 1.5,
 		α_p_dch = 1.5,
 		η_ch = 0.98,
 		η_dch = 0.98,
 		η_self = 0.0005,
 		α_soc_min = 0.2,
 		α_soc_max = 0.8,
 		lifetime = 12,
 		nCycle = 2500.,
 		bounds = (lb = 0., ub = 1000.),
 		SoH_threshold = 0.8,
 		couplage = (E = true, R = false),
 		soc_model = "linear",
        calendar = true,
 		Erated_ini = 1e-6,
 		soc_ini = 0.5,
 		soh_ini = 1.,
        artificial_soc_profil = zeros(8760,1)) =  verification_liion_params(α_p_ch, α_p_dch, η_ch, η_dch, η_self, α_soc_min, α_soc_max, lifetime, nCycle, bounds,
            SoH_threshold, couplage, soc_model, calendar, Erated_ini, soc_ini, soh_ini, artificial_soc_profil) ?
 			new(α_p_ch, α_p_dch, η_ch, η_dch, η_self, α_soc_min, α_soc_max, lifetime, nCycle, bounds,
 			SoH_threshold, couplage, soc_model, calendar, Erated_ini, soc_ini, soh_ini, artificial_soc_profil) : nothing

 end

### Preallocation
function preallocate!(liion::Liion_energy_exchanged, nh::Int64, ny::Int64, ns::Int64)
    liion.Erated = convert(SharedArray,zeros(ny+1, ns)) ; liion.Erated[1,:] .= liion.Erated_ini
    liion.carrier = Electricity()
    liion.carrier.power = convert(SharedArray,zeros(nh, ny, ns))
    if liion.soc_model == "artificial"
        liion.soc = convert(SharedArray,reshape(repeat(liion.artificial_soc_profil,ns), (nh+1,ny+1,ns)))
    else
        liion.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soc[1,1,:] .= liion.soc_ini
    end
    liion.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soh[1,1,:] .= liion.soh_ini
    liion.cost = convert(SharedArray,zeros(ny, ns))
    liion.voltage = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.voltage[1,1:2,:] .= 3.7
    liion.current = convert(SharedArray,zeros(nh+1, ny+1, ns))
    liion.tremblay_dessaint_params = Tremblay_dessaint_params()

    liion.nCycle *= 5 #cycle to failure calibrée pour 80%
    return liion
end


 ### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, liion::Liion_energy_exchanged, decision::Float64, Δh::Int64)

    #Cycle part
    if liion.soc_model == "artificial"
        liion.soh[h+1,y,s] = liion.soh[h,y,s] - (liion.Erated[y,s] * (abs(liion.soc[h+1,y,s] - liion.soc[h,y,s])))  / (2. * liion.nCycle * liion.Erated[y,s] * (liion.α_soc_max - liion.α_soc_min) )
    else
        liion.soc[h+1,y,s], liion.carrier.power[h,y,s] = compute_operation_soc_linear(liion, (Erated = liion.Erated[y,s], soc = liion.soc[h,y,s], soh = liion.soh[h,y,s]), decision, Δh)

    	power_dch, power_ch = get_power_flow(liion, (Erated = liion.Erated[y,s], soc = liion.soc[h,y,s], soh = liion.soh[h,y,s]), decision, Δh)

        liion.soh[h+1,y,s] = liion.soh[h,y,s] - (power_dch - power_ch) * Δh / (2. * liion.nCycle * (liion.α_soc_max - liion.α_soc_min) * liion.Erated[y,s])
    end

    #Calendar part
    if liion.calendar == true
        liion.soh[h+1,y,s] = liion.soh[h+1,y,s] - (1 - exp(- 4.14e-10 * 3600 * Δh))
    end

end

function compute_operation_dynamics(liion::Liion_energy_exchanged, state::NamedTuple{(:Erated, :soc, :soh), Tuple{Float64, Float64, Float64}}, decision::Float64, Δh::Int64)

	power_dch, power_ch = get_power_flow(liion, (Erated = state.Erated, soc = state.soc, soh = state.soh), decision, Δh)
	return power_dch + power_ch
end

 ### Investment dynamic

  ### Investment dynamic
  function compute_investment_dynamics!(y::Int64, s::Int64, liion::Liion_energy_exchanged, decision::Union{Float64, Int64})
      if liion.soc_model == "artificial"
          liion.Erated[y+1,s], _, liion.soh[1,y+1,s], liion.voltage[1,y+1,s] = compute_investment_dynamics(liion, (Erated = liion.Erated[y,s], soc = liion.soc[end,y,s], soh = liion.soh[end,y,s], voltage = liion.voltage[end,y,s]), decision)
      else
          liion.Erated[y+1,s], liion.soc[1,y+1,s], liion.soh[1,y+1,s], liion.voltage[1,y+1,s] = compute_investment_dynamics(liion, (Erated = liion.Erated[y,s], soc = liion.soc[end,y,s], soh = liion.soh[end,y,s], voltage = liion.voltage[end,y,s]), decision)
      end
  end


  
  function initialize_investments!(s::Int64, liion::Liion_energy_exchanged, decision::Union{Float64, Int64})
	liion.Erated[1,s] = decision
	liion.soc[1,1,s] = liion.soc_ini
	liion.soh[1,1,s] = liion.soh_ini
end

  function compute_investment_dynamics(liion::Liion_energy_exchanged, state::NamedTuple{(:Erated, :soc, :soh, :voltage), Tuple{Float64, Float64, Float64, Float64}}, decision::Union{Float64, Int64})
      if decision > 1e-2
          Erated_next = decision
          soc_next = liion.soc_ini
          soh_next =  1.
      else
          Erated_next = state.Erated
          soc_next = state.soc
          soh_next = state.soh
      end

 	 #TODO faire verifier
 	 if state.voltage != 0
 		 voltage_next = state.voltage
 	 else
 		voltage_next = liion.soc_ini*0.7 + 3.3
 	 end

      return Erated_next, soc_next, soh_next, voltage_next
  end



 function verification_liion_params(α_p_ch::Float64, α_p_dch::Float64, η_ch::Float64, η_dch::Float64, η_self::Float64,
 	α_soc_min::Float64, α_soc_max::Float64, lifetime::Int64, nCycle::Float64, bounds::NamedTuple{(:lb, :ub), Tuple{Float64, Float64}},
 	SoH_threshold::Float64, couplage::NamedTuple{(:E,:R), Tuple{Bool,Bool}}, soc_model::String, calendar::Bool, Erated_ini::Float64, soc_ini::Float64,
 	soh_ini::Float64, artificial_soc_profil::Array{Float64,2})

 	validation = true

 	# α_p_ch, α_p_dch
 	if bounds.lb > bounds.ub
 		error("Lower bound cannot be inferieur to upper bound")
 		validation = false
 	end

 	if α_soc_min > α_soc_max
 		error("Lower bound cannot be inferieur to upper bound (soc bounds)")
 		validation = false
 	end

 	if !(soc_model in soc_model_names)
 		error(soc_model ," is not an authorized Liion state of charge model. you need to pick one from the following list : ", soc_model_names)
 		validation = false
 	end

 	if η_ch < 0 || η_ch > 1
 		error("η_ch out of bound [0-1] with value :" , η_ch)
 		validation = false
 	end

 	if η_dch < 0 || η_dch > 1
 		error("η_dch out of bound [0-1] with value :" , η_dch)
 		validation = false
 	end

 	if η_self < 0 || η_self > 1
 		error("η_self out of bound [0-1] with value :" , η_self)
 		validation = false
 	end

 	if α_soc_min < 0 || α_soc_min > 1
 		error("α_soc_min out of bound [0-1] with value :" , α_soc_min)
 		validation = false
 	end

 	if α_soc_max < 0 || α_soc_max > 1
 		error("α_soc_max out of bound [0-1] with value :" , α_soc_max)
 		validation = false
 	end

 	if Erated_ini < 0
 		error("Erated_ini must be positive")
 		validation = false
 	end

 	if soc_ini < 0 || soc_ini > 1
 		error("soc_ini out of bound [0-1] with value :" , soc_ini)
 		validation = false
 	end

 	if soh_ini < 0 || soh_ini > 1
 		error("soh_ini out of bound [0-1] with value :" , soh_ini)
 		validation = false
 	end

 	if SoH_threshold  < 0 || SoH_threshold >= 1
 		error("SoH_threshold out of bound [0-1[ with value :" , SoH_threshold)
 		validation = false
 	end


 	return validation
 end
