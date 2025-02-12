include(joinpath(pwd(),"src","Genesys2.jl"))


####################################################
####################################################
####################################################
####################################################
####################################################
####### Proposition de correction ##################
###########  Pour le Barrage  ######################
####################################################
####################################################
####################################################
####################################################
####################################################

# Nouveau composant 
mutable struct Barrage <: AbstractStorage

    η_E_O::Float64 # la conversion m^3/Wh 
    η_O_E::Float64 # la conversion Wh/m^3
    volume_ini::Float64 # 
    soc_ini::Float64 #initial SoC
    lifetime::Int64 # Si on n'utilise pas la mécanique de SoH il faut renseigner une durée de vie.
    lim_debit_pompe::Float64 # 
    lim_debit_turbine::Float64 #
    surface::Float64  # surface en m^2

    # Variables pour stocker les données d'environnement


    volume::AbstractArray{Float64,2} # Le volume total de la retenu d'eau

	carrier::Electricity #Type of energy
    soc::AbstractArray{Float64,3} #3 dim matrix nh*ny*ns ∈ [0-1] 0 indiquant que la retenue d'eau est tarie et 1 qu'elle est prête à deborder

    cost::AbstractArray{Float64,2} 

    Barrage(;η_E_O = 0.8,
    η_O_E = 0.9,
    volume_ini = 1e-6,
    soc_ini = 0.5,
    lifetime = 40,
    lim_debit_pompe = 12,
    lim_debit_turbine = 20,
    surface = 1,
    ) = new(η_E_O, η_O_E, volume_ini, soc_ini, lifetime, lim_debit_pompe, lim_debit_turbine, surface) 

end


function preallocate!(barrage::Barrage, nh::Int64, ny::Int64, ns::Int64)
    barrage.volume = convert(SharedArray,zeros(ny+1, ns)) ; barrage.volume[1,:] .= barrage.volume_ini
    barrage.carrier = Electricity()
    barrage.carrier.power = convert(SharedArray,zeros(nh, ny, ns)) 


    barrage.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; barrage.soc[1,1,:] .= barrage.soc_ini
    barrage.cost = convert(SharedArray,zeros(ny, ns))
    
    return barrage
 end


 function initialize_investments!(s::Int64, barrage::Barrage, decision::Union{Float64, Int64})
    barrage.volume[1,s] = decision
    barrage.soc[1,1,s] = barrage.soc_ini
 end
 

 function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, barrage::Barrage, decision::Float64, Δh::Float64)

	barrage.soc[h+1,y,s], power_ch, power_dch = compute_operation_soc(barrage, h ,y ,s , decision, Δh)
	
	barrage.carrier.power[h,y,s] = power_ch + power_dch 

end



function compute_operation_dynamics(h::Int64, y::Int64, s::Int64, barrage::Barrage, decision::Float64, Δh::Float64)

	soc_next, power_ch, power_dch  = compute_operation_soc(barrage, h ,y ,s , decision, Δh)
	
	return power_ch + power_dch, soc_next

end



function compute_operation_soc(barrage::Barrage, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Float64)
	if decision >= 0 
		η = barrage.η_O_E 
	else
		η = barrage.η_E_O 
	end

	power_turbin = max(min(decision, η * barrage.soc[h,y,s] * barrage.volume[y,s] / Δh, barrage.lim_debit_turbine), 0.) # On limite par le débit et par la quantité disponnible.
	power_pompe = min(max(decision, -barrage.lim_debit_pompe), 0.) #On authorise à déborder mais on limite le débit

    soc =  barrage.soc[h,y,s] - (power_pompe * η + power_turbin / η) * Δh / barrage.volume[y,s]  
    
	return min(soc, 1.), power_pompe, power_turbin

end




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

    # Storages
    # barrage
   
end


function compute_investment_dynamics!(y::Int64, s::Int64, barrage::Barrage, decision::Union{Float64, Int64})
	barrage.volume[y+1,s], barrage.soc[1,y+1,s] = compute_investment_dynamics(barrage, (volume = barrage.volume[y,s], soc = barrage.soc[end,y,s]), decision, s)
 end


 
function compute_investment_dynamics(barrage::Barrage, state::NamedTuple{(:volume, :soc), Tuple{Float64, Float64}}, decision::Union{Float64, Int64}, s::Int64)
    if decision > 1e-2
        volume_next = decision
        soc_next = barrage.soc_ini
    else
        volume_next = state.volume
        soc_next = state.soc
    end

    return volume_next, soc_next
end


# RB pour la gestion du barrage
function RB_barrage(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]
end


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
        return RB_barrage(h, y, s, mg, controller)
    else
        println("Policy not defined !")
    end
end




nh, ny, ns = 8760, 10, 1

pygui(true)
plotlyjs()

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Barrage(), 
                Grid(carrier = Electricity()))


using JLD2, FileIO

data_cours = JLD2.load(joinpath("Cours", "Cours4", "Data_base_TP4.jld2"))

seed = zeros(ny,ns)
seed[1:10,1] = [i for i in 1:10]
ω = Scenarios(microgrid, data_cours; same_year=false, seed=seed)
# Ici ω. + tab vous montre le champs du scénarios, ce sont les même que pour la structure microgrid et ils contiennent les données relatives aux éléments du microgrid.
# Ici ω.storages[1]. + tab vous montre le champs contenu dans les données relatives au barrage (ce sont les même que vous avez inclus dans la banque de données)


generations = Dict("Solar" => 40.)
storages = Dict( "Barrage" => 100.)
subscribed_power = Dict("Electricity" => 10.)
                
designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω)

controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = 104)), ω)


simulate!(microgrid, controller, designer, ω, options = Options(mode = "serial"))

metrics = Metrics(microgrid, designer)
    
plot_operation2(microgrid, y=1:ny, s=1:1)




# Sachant que l'évaportation en volume peut être calculée comme 0.006 * Temperature à 2m + 000.6 * irradiantion + 000.2 * vitesse vent
# Utilisez les fichier  vent.csv ,TLSE_prod_PV_2005_2020.csv et pluie.csv pour ajouter un model d'évaporation

# Vous devez recuperer les données dans les 3 fichier, 
# modifier la création de la banque de donnée pour y inserer les données de vent

# Modifier les fonctions de mise à jour du barrage et y inclure une diminution du volume en utilisant l'équation d'évaporation fournie
        # Pour cela vous devez : - récuperer les données concernées et les stocker dans la structure barrage via update_operation_informations
        #                        - modifier 
        
# mutable struct Barrage <: AbstractStorage
# function preallocate!(barrage::Barrage, nh::Int64, ny::Int64, ns::Int64)
# function get_pluie(rain::Float64)
# function get_evaporation(irradiance::Float64, wind::Float64, temperature::Float64)
# function update_operation_informations!(h::Int64, y::Int64, s::Int64, mg::Microgrid, ω::Scenarios)


# Charger les données de scénario et testez votre évaporation











####################################################
####################################################
####################################################
####################################################
####################################################
####### Proposition de correction ##################
###########  Pour le TP4  ######################
####################################################
####################################################
####################################################
####################################################
####################################################
include(joinpath(pwd(),"src","Genesys2.jl"))

# Nouveau composant 
mutable struct Barrage <: AbstractStorage

    η_E_O::Float64 # la conversion m^3/Wh 
    η_O_E::Float64 # la conversion Wh/m^3
    volume_ini::Float64 # 
    soc_ini::Float64 #initial SoC
    facteur_pluie::Float64 # La part maximal qui peut être remplie par la pluie durant 1 pas de temps
    facteur_evaporation::Float64 # La part maximal qui peut être évaporée durant 1 pas de temps
    lifetime::Int64 # Si on n'utilise pas la mécanique de SoH il faut renseigner une durée de vie.
    lim_debit_pompe::Float64 # 
    lim_debit_turbine::Float64 #
    surface::Float64  # surface en m^2

    # Variables
    irradiance::AbstractArray{Float64,3} #l'ensoleillement qui va alimenter l'evaporation
    wind::AbstractArray{Float64,3} #le vent qui va alimenter l'evaporation
    temperature::AbstractArray{Float64,3} #la temperature qui va alimenter l'évaporation
    rain::AbstractArray{Float64,3} #la pluie qui va alimenter le remplissage


    volume::AbstractArray{Float64,2} # Le volume total de la retenu d'eau

	carrier::Electricity #Type of energy
    soc::AbstractArray{Float64,3} #3 dim matrix nh*ny*ns ∈ [0-1] 0 indiquant que la retenue d'eau est tarie et 1 qu'elle est prête à deborder

    cost::AbstractArray{Float64,2} 

    Barrage(;η_E_O = 0.8,
    η_O_E = 0.9,
    volume_ini = 1e-6,
    soc_ini = 0.5,
    facteur_pluie = 1/100,
    facteur_evaporation = 1/500,
    lifetime = 40,
    lim_debit_pompe = 12,
    lim_debit_turbine = 20,
    surface = 1,
    ) = new(η_E_O, η_O_E, volume_ini, soc_ini, facteur_pluie, facteur_evaporation, lifetime, lim_debit_pompe, lim_debit_turbine, surface) 

end


function preallocate!(barrage::Barrage, nh::Int64, ny::Int64, ns::Int64)
    barrage.volume = convert(SharedArray,zeros(ny+1, ns)) ; barrage.volume[1,:] .= barrage.volume_ini
    barrage.carrier = Electricity()
    barrage.carrier.power = convert(SharedArray,zeros(nh, ny, ns)) 

    barrage.irradiance = convert(SharedArray,zeros(nh, ny, ns))  
    barrage.wind = convert(SharedArray,zeros(nh, ny, ns))  
    barrage.temperature = convert(SharedArray,zeros(nh, ny, ns))  
    barrage.rain = convert(SharedArray,zeros(nh, ny, ns))  

    barrage.soc = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; barrage.soc[1,1,:] .= barrage.soc_ini
    barrage.cost = convert(SharedArray,zeros(ny, ns))
    
    return barrage
 end


 function initialize_investments!(s::Int64, barrage::Barrage, decision::Union{Float64, Int64})
    barrage.volume[1,s] = decision
    barrage.soc[1,1,s] = barrage.soc_ini
 end
 

 function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, barrage::Barrage, decision::Float64, Δh::Float64)

	barrage.soc[h+1,y,s], power_ch, power_dch = compute_operation_soc(barrage, h ,y ,s , decision, Δh)
	
	barrage.carrier.power[h,y,s] = power_ch + power_dch 

end



function compute_operation_dynamics(h::Int64, y::Int64, s::Int64, barrage::Barrage, decision::Float64, Δh::Float64)

	soc_next, power_ch, power_dch  = compute_operation_soc(barrage, h ,y ,s , decision, Δh)
	
	return power_ch + power_dch, soc_next

end



function compute_operation_soc(barrage::Barrage, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Float64)
	if decision >= 0 
		η = barrage.η_O_E 
	else
		η = barrage.η_E_O 
	end

    evaporation = get_evaporation(barrage, h, y ,s)   
    pluie = get_pluie(barrage.rain[h,y,s]) * barrage.surface   # Remplissage due à la pluie 


	power_turbin = max(min(decision, η * barrage.soc[h,y,s] * barrage.volume[y,s] / Δh, barrage.lim_debit_turbine), 0.) # On limite par le débit et par la quantité disponnible.
	power_pompe = min(max(decision, -barrage.lim_debit_pompe), 0.) #On authorise à déborder mais on limite le débit

    soc = ((pluie-evaporation)/barrage.volume[y,s]) +  barrage.soc[h,y,s] - (power_pompe * η + power_turbin / η) * Δh / barrage.volume[y,s]  
    
	return min(soc, 1.), power_pompe, power_turbin

end




############### Ajout de la pluie et de l'évaporation ##############################
function get_pluie(rain::Float64)

    return rain

end


function get_evaporation(barrage::Barrage, h::Int64, y::Int64, s::Int64)
    return min(0.006 * barrage.temperature[h,y,s] + 0.006 * barrage.irradiance[h,y,s] + 0.002 * barrage.wind[h,y,s], barrage.soc[h,y,s] * barrage.volume[y,s])
end


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

    # barrage
    for (k, a) in enumerate(mg.storages)
        if a isa Barrage
            a.irradiance[h,y,s] = ω.storages[k].irradiance[h,y,s]
            a.rain[h,y,s] =  ω.storages[k].rain[h,y,s]
            a.wind[h,y,s] =  ω.storages[k].wind[h,y,s]
        end
    end

end


function compute_investment_dynamics!(y::Int64, s::Int64, barrage::Barrage, decision::Union{Float64, Int64})
	barrage.volume[y+1,s], barrage.soc[1,y+1,s] = compute_investment_dynamics(barrage, (volume = barrage.volume[y,s], soc = barrage.soc[end,y,s]), decision, s)
 end


 
function compute_investment_dynamics(barrage::Barrage, state::NamedTuple{(:volume, :soc), Tuple{Float64, Float64}}, decision::Union{Float64, Int64}, s::Int64)
    if decision > 1e-2
        volume_next = decision
        soc_next = barrage.soc_ini
    else
        volume_next = state.volume
        soc_next = state.soc
    end

    return volume_next, soc_next
end


# RB pour la gestion du barrage
function RB_barrage(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]
end


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
        return RB_barrage(h, y, s, mg, controller)
    else
        println("Policy not defined !")
    end
end




nh, ny, ns = 8760, 10, 1

pygui(true)
plotlyjs()

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Barrage(; facteur_pluie= 5/4, facteur_evaporation = 1/5), 
                Grid(carrier = Electricity()))


using JLD2, FileIO

data_cours = JLD2.load(joinpath("Cours", "Cours4", "Data_base_TP4.jld2"))

ω = Scenarios(microgrid, data_cours; same_year=true, seed=[x for x in 1:ns])

generations = Dict("Solar" => 40.)
storages = Dict( "Barrage" => 100.)
subscribed_power = Dict("Electricity" => 10.)
                
designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω)

controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = 104)), ω)


simulate!(microgrid, controller, designer, ω, options = Options(mode = "serial"))

metrics = Metrics(microgrid, designer)
    
plot_operation2(microgrid, y=1:ny, s=1:1)

