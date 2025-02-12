
# Chargement du package
include(joinpath(pwd(),"src","Genesys2.jl"))

# Redéfinition de 2 fonctions relatives au control Rule Based et aux scénarios
function compute_operation_decisions!(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Chose policy
    if controller.options.policy_selection == 1
        return π_1(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 2
        return π_2(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 3
        return π_3(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 4
        return π_4(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 5
        return π_5(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 6
        return π_6(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 7
        return π_7(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 101
        return RB_autonomie(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 102
        return RB_vieillissement(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 103
        return RB_opex(h, y, s, mg, controller)
    elseif controller.options.policy_selection == 104
        return RB_barage(h, y, s, mg, controller)
    else
        println("Policy not defined !")
    end
end



############### Ajout de la pluie et de l'évaporation ##############################
#Cette fonction ainsi que celle d'évaporation sont purement à but pédagogique et ne sont basées sur aucune données.
# La pluie comme une variable aléatoire dépendante de la production PV.
function get_pluie(ensoleillement::Float64, facteur::Float64, h::Int64)

    isItNight = h%24 < 7 || h%24 > 21  # On regarde si c'est la nuit

    if ensoleillement <= 0.05 && !isItNight # Si l'ensoleillement est presque nul et que ce n'est pas la nuit 
        # On utilise une distribution de pluie
        α = 2
        θ = 15
        dist = Gamma(α,θ)
        val = rand(dist) 

        #figure("ensoleillement très faible de jour")
        #Seaborn.kdeplot(  [rand(truncated(dist,9,100)) for i in 1:10000] )
        
    elseif ensoleillement <= 0.2 && !isItNight# Sinon, si l'ensoleillement est faible et que ce n'est pas la nuit 
        # On utilise une distribution de pluie dont l'esperance est plus faible
        α = 1.5
        θ = 2
        dist = Gamma(α,θ)
        val = rand(dist) 
        #figure("ensoleillement faible de jour")
        #Seaborn.kdeplot(  [rand(dist) for i in 1:10000] )

    elseif isItNight # Sinon si cest la nuit on a une autre distribution
        θ = 3
        dist = Exponential(θ)
        val = rand(dist)
        
        #figure("nuit")
        #Seaborn.kdeplot(  [rand(dist) for i in 1:10000] )

    else # Il faut trop soleil alors il ne pleut pas
        val = 0
    end

    return  min(val * facteur, 50)

    # μ = 0
    # σ = 1
    # ξ = 0.03
    # dist = GeneralizedPareto(μ, σ, ξ) 
    # rand(dist) 
    # values = [rand(dist) for i in 1:10000]
    # Seaborn.hist(values, bins=50)

end

# L'évaporation en fonction de la production PV

function get_evaporation(ensoleillement::Float64, facteur::Float64)

    return ensoleillement * facteur
end



#################################################
####### Votre implémentation du barage ##########
################################################



#################################################







































####################################################
####################################################
####################################################
####################################################
####################################################
####### Proposition de correction ##################
####################################################
####################################################
####################################################
####################################################
####################################################
####################################################

abstract type AbstractBarageAgingModel end

mutable struct FixedLifetimeBarrage <: AbstractBarageAgingModel

	lifetime::Int64

	FixedLifetimeBarrage(;lifetime = 40) = new(lifetime)
end

# Nouveau composant 
mutable struct Barrage <: AbstractStorage

    SoH_model::AbstractBarageAgingModel
    η_E_O::Float64 # la conversion m^3/Wh 
    η_O_E::Float64 # la conversion Wh/m^3
    volume_ini::Float64 # 
    soc_ini::Float64 #initial SoC
    soh_ini::Float64 #initial SoC
    SoH_threshold::Float64 # limit for replacement
    facteur_pluie::Float64 # Facteur arbitraire relatif au remplissage par précipitation
    facteur_evaporation::Float64 # Facteur arbitraire relatif  àl'évaporation
    lifetime::Int64 # Si on n'utilise pas la mécanique de SoH il faut renseigner une durée de vie.
    lim_debit_pompe::Int64 # Limite en puissance de pompage
    lim_debit_turbine::Int64 # Limite en puissance de turbinage


    # Variables
    ensoleillement::AbstractArray{Float64,3} #l'ensoleillement qui va alimenter une fonction de probabilité pour l'évaporation ou le remplissage
    volume::AbstractArray{Float64,2} # Le volume total de la retenu d'eau
	carrier::Electricity #Type of energy
    soc::AbstractArray{Float64,3} #3 dim matrix nh*ny*ns ∈ [0-1] 0 indiquant que la retenue d'eau est tarie et 1 qu'elle est prête à deborder
    soh::AbstractArray{Float64,3} #3 dim matrix nh*ny*ns ∈ [0-1] 0 indiquant que la retenue d'eau est tarie et 1 qu'elle est prête à deborder

    cost::AbstractArray{Float64,2} 



    Barrage(;SoH_model = FixedLifetimeBarrage(),
    η_E_O = 0.8,
    η_O_E = 0.9,
    volume_ini = 1e-6,
    soc_ini = 0.5,
    soh_ini = 1.,
    SoH_threshold = .2,
    facteur_pluie = 1/100,
    facteur_evaporation = 1/500,
    lifetime = 40,
    lim_debit_pompe = 20,
    lim_debit_turbine = 25,
    ) = new(SoH_model, η_E_O, η_O_E, volume_ini, soc_ini, soh_ini, SoH_threshold, facteur_pluie, facteur_evaporation, lifetime, lim_debit_pompe, lim_debit_turbine) 

end

# Alloue la mémoire aux différents tableau de la structures et initialise certaines valeurs
function preallocate!(barage::Barrage, nh::Int64, ny::Int64, ns::Int64)
    barage.volume = convert(SharedArray,zeros(ny+1, ns)) ; barage.volume[1,:] .= barage.volume_ini
    barage.carrier = Electricity()
    barage.carrier.power = convert(SharedArray,zeros(nh, ny, ns))  
    barage.ensoleillement = convert(SharedArray,zeros(nh, ny, ns))  

    barage.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; barage.soc[1,1,:] .= barage.soc_ini
    barage.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; barage.soh[1,1,:] .= barage.soh_ini

    barage.cost = convert(SharedArray,zeros(ny, ns))
    
    return barage
 end

# Met le composant en état initial
 function initialize_investments!(s::Int64, barage::Barrage, decision::Union{Float64, Int64})
    barage.volume[1,s] = decision
    barage.soc[1,1,s] = barage.soc_ini
    barage.soh[1,1,s] = barage.soh_ini

 end
 
# Function used to apply decisions
 function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, barage::Barrage, decision::Float64, Δh::Float64)

	barage.soc[h+1,y,s], power_ch, power_dch = compute_operation_soc(barage, h ,y ,s , decision, Δh)
	
	barage.carrier.power[h,y,s] = power_ch + power_dch 

    barage.soh[h+1,y,s] = compute_operation_soh(barage, h ,y ,s, Δh)
end


#Function used to test decisions (for RB control)
function compute_operation_dynamics(h::Int64, y::Int64, s::Int64, barage::Barrage, decision::Float64, Δh::Float64)

	soc_next, power_ch, power_dch  = compute_operation_soc(barage, h ,y ,s , decision, Δh)
	
	return power_ch + power_dch, soc_next

end


# Fonction de calcul de la nouvelle valeur de SoC prenant en compte la pluie, l'évaporation, le pompage et le turbinage
function compute_operation_soc(barage::Barrage, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Float64)
	if decision >= 0 
		η = barage.η_O_E 
	else
		η = barage.η_E_O 
	end

    evaportation = get_evaporation(barage.ensoleillement[h,y,s], barage.facteur_evaporation)    # Evaporation due à la chaleur
    pluie = get_pluie(barage.ensoleillement[h,y,s], barage.facteur_pluie, h)    # Remplissage due à la pluie 


	power_turbin = max(min(decision, η * barage.soc[h,y,s] * barage.volume[y,s] / Δh, barage.lim_debit_turbine), 0.) # On limite par le débit et par la quantité disponnible.
	power_pompe = min(max(decision, -barage.lim_debit_pompe), 0.) #On authorise à déborder mais on limite le débit

    soc = (pluie/barage.volume[y,s] ) + (1- evaportation) * barage.soc[h,y,s] - (power_pompe * η + power_turbin / η) * Δh / barage.volume[y,s]  
    
	return min(soc, 1.), power_pompe, power_turbin

end

#Fonction de calcul de la nouvelle valeur du SoH
function compute_operation_soh(barage::Barrage, h::Int64, y::Int64, s::Int64, Δh::Float64)

    return barage.soh[h,y,s] - (Δh / (barage.lifetime * 8760))

end

# Mise à jour de la fonction de récupération des données (activée dans simulate, à chaque pas de temps)
function update_operation_informations!(h::Int64, y::Int64, s::Int64, mg::Microgrid, ω::Scenarios)
    # Demands
    for (k, a) in enumerate(mg.demands)
        a.timestamp[h,y,s] = ω.demands[k].t[h,y,s]
        a.carrier.power[h,y,s] = ω.demands[k].power[h,y,s]
    end
    # Generations
    for (k, a) in enumerate(mg.generations)
        a.timestamp[h,y,s] = ω.generations[k].t[h,y,s]
        a.carrier.power[h,y,s] = a.powerMax[y,s] * ω.generations[k].power[h,y,s]
    end

    # barage
    for (k, a) in enumerate(mg.storages)
        if a isa Barrage
            a.ensoleillement[h,y,s] = ω.generations[k].power[h,y,s]
        end
    end

end

# Fonction de remplacement du barage (si la decision de remplacement est prise)
function compute_investment_dynamics!(y::Int64, s::Int64, barage::Barrage, decision::Union{Float64, Int64})
	barage.volume[y+1,s], barage.soc[1,y+1,s], barage.soh[1,y+1,s] = compute_investment_dynamics(barage, (volume = barage.volume[y,s], soc = barage.soc[end,y,s], soh = barage.soh[end,y,s]), decision, s)
 end


# Fonction de remplacement du barage (si la decision de remplacement est prise)
function compute_investment_dynamics(barage::Barrage, state::NamedTuple{(:volume, :soc, :soh), Tuple{Float64, Float64, Float64}}, decision::Union{Float64, Int64}, s::Int64)
    if decision > 1e-2
        volume_next = decision
        soc_next = barage.soc_ini
        soh_next = barage.soh_ini
    else
        volume_next = state.volume
        soc_next = state.soc
        soh_next = state.soh
    end

    return volume_next, soc_next, soh_next
end


# RB pour la gestion du barage
function RB_barage(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]
end




nh, ny, ns = 8760, 10, 4

pygui(true)

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Barrage(; facteur_pluie= 5/4, facteur_evaporation = 1/5), 
                Grid(carrier = Electricity()))


using JLD2, FileIO

data_optim = JLD2.load(joinpath(pwd(),"Cours", "cours4", "Data_base_TP4.jld2"))
            
ω_a = Scenarios(microgrid, data_optim, true)
            
generations = Dict("Solar" => 30.)
storages = Dict( "Barrage" => 400.)
subscribed_power = Dict("Electricity" => 10.)
                


designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω_a)

controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = 104)), ω_a)

simulate!(microgrid, controller, designer, ω_a, options = Options(mode = "serial"))

#metrics = Metrics(microgrid, designer)
    
plot_operation(microgrid, y=1:ny, s=1:1)


