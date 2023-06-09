

# Ce modèle à beaucoup de code inutile car il a été codé dans la précipitation à partir d"un copié collé,
#Il mérite d'être réécrit.
#=
    Li-ion battery modelling
 =#

 """
 	Liion_fixed_lifetime

A mutable struct that represents a Li-ion battery model with a linear calendar aging for State of Health (SoH) computation.

The structure have a lot of parameters but most of them are set to default values.

# Parameters:
- `α_p_ch::Float64`: Charging maximum C-rate (default : 1.5)
- `α_p_dch::Float64`: Discharging  maximum C-rate (default : 1.5)
- `α_soc_min::Float64`: Minimum threshold of charge (normalized) (default : 0.2)
- `α_soc_max::Float64`: Maximum threshold of charge (normalized) (default : 0.8)
- `SoH_threshold::Float64`: SoH level to replace the battery (default : 0.8)
- `couplage::NamedTuple`: Named tuple with two boolean values to indicate if the SoH should influence the other parameters (E stand for capacity coupling and R for efficiency coupling)
- `soc_model::String`: Model name for State of Charge (SoC) computation. Available models are listed 
- `soc_ini::Float64`: Initial State of Charge (SoC) for the beginning of the simulation (default : 0.5)
- `soh_ini::Float64`: Initial State of Health (SoH) for the beginning of the simulation (default : 1)

## Example 
```julia
Liion_fixed_lifetime(;soc_model = "polynomial", couplage = (E=true, R=true))
```

"""
 mutable struct Liion_fixed_lifetime <: AbstractLiion


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

 	# Initial conditions
 	Erated_ini::Float64  # capacité de la batterie en Wh
 	soc_ini::Float64 # first state of charge for the begining of simulation
 	soh_ini::Float64 # first state of health for the begining of simulation


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
 	Liion_fixed_lifetime(; α_p_ch = 1.5,
 		α_p_dch = 1.5,
 		η_ch = 0.9,
 		η_dch = 0.9,
 		η_self = 0.0005,
 		α_soc_min = 0.2,
 		α_soc_max = 0.8,
 		lifetime = 12,
 		nCycle = 2500.,
 		bounds = (lb = 0., ub = 1000.),
 		SoH_threshold = 0.8,
 		couplage = (E = true, R = false),
 		soc_model = "linear",
 		Erated_ini = 1e-6,
 		soc_ini = 0.5,
 		soh_ini = 1.) =  verification_liion_params(α_p_ch, α_p_dch, η_ch, η_dch, η_self, α_soc_min, α_soc_max, lifetime, nCycle, bounds,
            SoH_threshold, couplage, soc_model, Erated_ini, soc_ini, soh_ini) ?
 			new(α_p_ch, α_p_dch, η_ch, η_dch, η_self, α_soc_min, α_soc_max, lifetime, nCycle, bounds,
 			SoH_threshold, couplage, soc_model, Erated_ini, soc_ini, soh_ini) : nothing

 end

### Preallocation
function preallocate!(liion::Liion_fixed_lifetime, nh::Int64, ny::Int64, ns::Int64)
    liion.Erated = convert(SharedArray,zeros(ny+1, ns)) ; liion.Erated[1,:] .= liion.Erated_ini
    liion.carrier = Electricity()
    liion.carrier.power = convert(SharedArray,zeros(nh, ny, ns))
    liion.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soc[1,1,:] .= liion.soc_ini
    liion.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.soh[1,1,:] .= liion.soh_ini
    liion.cost = convert(SharedArray,zeros(ny, ns))
    liion.voltage = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; liion.voltage[1,1:2,:] .= 3.7
    liion.current = convert(SharedArray,zeros(nh+1, ny+1, ns))
    liion.tremblay_dessaint_params = Tremblay_dessaint_params()
    return liion
end


 ### Operation dynamic
function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, liion::Liion_fixed_lifetime, decision::Float64, Δh::Int64)
    #Cycle part

        liion.soc[h+1,y,s], liion.carrier.power[h,y,s] = compute_operation_soc_linear(liion, (Erated = liion.Erated[y,s], soc = liion.soc[h,y,s], soh = liion.soh[h,y,s]), decision, Δh)

        liion.soh[h+1,y,s] = liion.soh[h,y,s] - ((1 - liion.SoH_threshold) * Δh)/(8760 * liion.lifetime)

     if y >1 && (y-1) % liion.lifetime == 0 && h == 8760
         liion.soh[h+1,y,s] = 0
     end
end

# function compute_operation_dynamics(liion::Liion_fixed_lifetime, state::NamedTuple{(:Erated, :soc, :soh), Tuple{Float64, Float64, Float64}}, decision::Float64, Δh::Int64)
#
#      if liion.couplage.E
#          Erated = state.Erated * state.soh
#      else
#          Erated = state.Erated
#      end
#
#      # Control power constraint and correction
#      power_dch = max(min(decision, liion.α_p_dch * Erated, state.soh * Erated / Δh, liion.η_dch * (state.soc * (1. - liion.η_self * Δh) - liion.α_soc_min) * Erated / Δh), 0.)
#      power_ch = min(max(decision, -liion.α_p_ch * Erated, -state.soh * Erated / Δh, (state.soc * (1. - liion.η_self * Δh) - liion.α_soc_max) * Erated / Δh / liion.η_ch), 0.)
#      # SoC dynamic
#      soc_next = state.soc * (1. - liion.η_self * Δh) - (power_ch * liion.η_ch + power_dch / liion.η_dch) * Δh / Erated
#      # SoH dynamic
#
#      return soc_next, soh_next, power_dch + power_ch
# end

 ### Investment dynamic

  ### Investment dynamic
  function compute_investment_dynamics!(y::Int64, s::Int64, liion::Liion_fixed_lifetime, decision::Union{Float64, Int64})
      liion.Erated[y+1,s], liion.soc[1,y+1,s], liion.soh[1,y+1,s], liion.voltage[1,y+1,s] = compute_investment_dynamics(liion, (Erated = liion.Erated[y,s], soc = liion.soc[end,y,s], soh = liion.soh[end,y,s], voltage = liion.voltage[end,y,s]), decision)
  end

  function compute_investment_dynamics(liion::Liion_fixed_lifetime, state::NamedTuple{(:Erated, :soc, :soh, :voltage), Tuple{Float64, Float64, Float64, Float64}}, decision::Union{Float64, Int64})
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
 	SoH_threshold::Float64, couplage::NamedTuple{(:E,:R), Tuple{Bool,Bool}}, soc_model::String, Erated_ini::Float64, soc_ini::Float64,
 	soh_ini::Float64)

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
