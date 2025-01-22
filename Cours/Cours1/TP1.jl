include(joinpath(pwd(),"src","Genesys2.jl"))


pyplot_installed = true
if !(isdir(Pkg.dir("PyPlot")))
    Pkg.add("plotlyjs")
    using plotlyjs
    plotlyjs()
    pyplot_installed = false
else
    using PyPlot
    pygui(true)
end



function Plot_dayly_prices(ω; label = "")
    
    if !pyplot_installed
        p = Plots.plot(1:24, ω.grids[1].cost_in[1:24,1,1],seriestype=:scatter, label = label)
        xlabel!("Hours")
        ylabel!("Price (€)")
        display(p)
    else
        PyPlot.scatter(1:24, ω.grids[1].cost_in[1:24,1,1], label = label)
        PyPlot.xlabel("Hours")
        PyPlot.ylabel("Price (€)")
        legend()
    end
  
end


#La place de ce code est normalement dans le fichier src/optimization/controller/rb.jl 
#Redefinition de la fonction de selection de Rule Base controller pour ajouter 101, 102, 103

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
    else
        println("Policy not defined !")
    end
end

#Il faut ici déterminer, en fonction des inputs (load, generation), les décisions pour les éléments de stockage et de conversion
#Il faut affecter les valeurs pour le pas de temps donné aux variables de décisions.
# P^{PV}_h est accéssible dans la variable    mg.generations[1].carrier.power[h,y,s]
# P^{load}_h est accéssible dans la variable   mg.demands[1].carrier.power[h,y,s]

#Décisions négative = charge, Décision positive = décharge (car on se place du point de vue de notre microréseau)

function RB_autonomie(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC) # policy 101
   #controller.decisions.storages[1][h,y,s] = calcul de votre décision
end

function RB_vieillissement(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC) # policy 102
    #controller.decisions.storages[1][h,y,s] = calcul de votre décision
end

function RB_opex(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC) # policy 103
    #controller.decisions.storages[1][h,y,s] = calcul de votre décision
end



# 1 scénario de 1 année avec 8760 pas de temps, donc un pas horaire.
nh, ny, ns = 8760, 1, 4

pygui(true)

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion(eff_model = PolynomialLiionEfficiency(), SoH_model = SemiEmpiricalLiion()),
                Grid(carrier = Electricity()))
                

using JLD2, FileIO

# Load data
data_fix = JLD2.load(joinpath(pwd(), "Cours" , "Cours1", "data_light_4.jld2"))
data_HP_HC = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4_HP_HC.jld2"))

data_selected = data_HP_HC
     
# Load data Scenarios
ω_a = Scenarios(microgrid, data_selected, true, seed=[x for x in 1:4])

Plot_dayly_prices(ω_a; label = "HP HC")



############# Dimentionnement manuel du réseau #####################
generations = Dict("Solar" => 0.) # Scalar Value for PV panel peak power 
storages = Dict("Liion" => 0.) # Scalar value for Li-ion battery capacity [kwh]
subscribed_power = Dict("Electricity" => 20.) # Maximum available power from grid [kVA]

designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω_a)
###################################################################


############# Controle du réseau ##################################
RB_choisie = 101 # 101, 102 ou 103

controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = RB_choisie)), ω_a)
###################################################################

############## Evaluation, métriques et affichage ###############
using BenchmarkTools
@benchmark simulate!(microgrid, controller, designer, ω_a, options = Options(mode = "serial"))

metrics = Metrics(microgrid, designer)
# Cout d'operation
println("Cout d'opération : ", round.(metrics.cost.opex[1,1:ns], digits=2), " €")
# Part de renouvelable
println("Part de renouvelable : ", round.(metrics.renewable_share[1,1:ns] * 100, digits=2), " %")
# Cout total (investissement compris)
println("Cout total : ", round.(metrics.cost.total[:,1:ns], digits=2), " €")

    
if pyplot_installed
    plot_operation(microgrid, y=1:ny, s=1:1)
else
    plot_operation2(microgrid, y=1:ny, s=1:1)
end
###################################################################






##########################################################################
##########################################################################
##########################################################################
##################### Les solutions du TP ###############################
##########################################################################
##########################################################################
##########################################################################




############################ Les Rule Base ###############################

function RB_autonomie(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Correction possible pour RB RB_autonomie
    # On favorise l'autoconsomation en chargeant la batterie avec le surplus de la différence (generation - demande)
    # Dans le cas où au contraire la demande est supérieur à la production on demande la différence à la batterie.
    # la fonction compute_power_balance s'occupe de calculer les variables de recours au cas où les décisions conduiraient
    # à des situations interdites
    controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]
end

function RB_vieillissement(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Correction possible pour RB RB_vieillissement
    # On choisi simplement de ne pas utiliser la batterie
    controller.decisions.storages[1][h,y,s] = 0
end

#Décisions négative = charge, Décision positive = décharge
function RB_opex(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Correction possible pour RB opex version 1
    # Si l'on se trouve en heure creuse on charge 6% dans la batterie en espérant rentabiliser plus tard cette énergie.
    # Sinon on utilise la différence entre la demande et la génération pour calculer l'opération
    if h%24 < 6 || h%24 > 21
        controller.decisions.storages[1][h,y,s] = -0.06 * mg.storages[1].Erated[y,s]
    else
        controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]
    end

end

#Décisions négative = charge, Décision positive = décharge
function RB_opex(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    # Correction possible pour RB opex version 2

    # Cette méthode est plus complèxe que l'autre solution mais montre plus de manipulation.
    # L'idée ici est de calculer l'opération en fonction de la différence (demand - generation).
    # On regarde ensuite 2 choses, si l'on est en heure creuse et si la batterie sera remplie par la commande
    # Si la batterie ne sera pas pleine alors modifie la commande pour demander de recharger la batterie de 10%


    liion = mg.storages[1]
    p_net_E = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]

    p_liion, soc_liion = compute_operation_dynamics(liion, h, y, s, p_net_E, mg.parameters.Δh)

    if h%24 < 6 || h%24 > 21
        if soc_liion <= liion.α_soc_max
            controller.decisions.storages[1][h,y,s] = p_liion - liion.Erated[y,s]/10
        else
            controller.decisions.storages[1][h,y,s] = p_liion
        end
    else
        controller.decisions.storages[1][h,y,s] = p_liion
    end

end

#################### renewable share #############################


# renvoi une valeur moyenne  sur tout les scenarios, années, heures de la part renouvelable
function compute_RES1(mg::Microgrid)
    grid_tot = sum(max.(0. ,mg.grids[1].carrier.power)) #Somme des puissances (non négatives) demandées au réseau 
    load_tot = sum(mg.demands[1].carrier.power) #Somme de la charge 
    return 1. - (grid_tot / load_tot) # On calcul le complémentaire de la part du réseau dans la satisfaction de la demande.
end


# renvoi une valeur moyenne sur toutes les années et heures de la part renouvelable. Une valeur par scénario donc.
function compute_RES2(mg::Microgrid)
    grid_tot = sum(max.(0. ,mg.grids[1].carrier.power); dims = (1,2)) #Somme sur les heures et années des puissances (non négatives) demandées au réseau 
    load_tot = sum(mg.demands[1].carrier.power; dims = (1,2)) #Somme sur les heures et années de la charge 
    return dropdims(1. .- (grid_tot ./ load_tot), dims = (1,2) ) # On calcul le complémentaire de la part du réseau dans la satisfaction de la demande.
    #Comme on a fait la somme sur les heures et années, on a un résultat par scénario.
    #la fonction dropdims nous permet de faire disparaitre les dimention des heures et années qui à été réduites via la somme.
    #Vous pouvez constater le changement de dimention grâce à dropdims passant de 1*1*ns à ns
    #Si vous faites varier ns vous obtiendrez un résultat dans les dimentions correspondantes.
end


# renvoi une valeur moyenne sur toutes les heures de chaque année de la part renouvelable. Une valeur par scénario et par an
function compute_RES3(mg::Microgrid)
    grid_tot = sum(max.(0. ,mg.grids[1].carrier.power) ; dims = 1) #Somme sur les heures des puissances (non négatives) demandées au réseau 
    load_tot = sum(mg.demands[1].carrier.power; dims =1) #Somme sur les heures de la charge 
    return dropdims(1. .- (grid_tot ./ load_tot), dims=1) # On calcul le complémentaire de la part du réseau dans la satisfaction de la demande.
    #Comme on a fait la somme sur les heures, on a un résultat par an et par scénario.
    #la fonction dropdims nous permet de faire disparaitre la dimention des heures qui à été réduite via la somme.
    #Vous pouvez constater le changement de dimention grâce à dropdims passant de 1*ny*ns à ny*ns
    #Si vous faites varier ny et ns vous obtiendrez un résultat dans les dimentions correspondantes.
end

###################################################################


####################### Vieillissement ###############################

#Retourne l'état de santé final du composant batterie
function compute_aging(mg::Microgrid)
    return mg.storages[1].soh[1,end,:] # Le mot 'end' permet ici d'acceder au dernier indice de la dimention. 
    # On notera qu'une année ny+1 existe dans la structure dont seule la premiere heure est remplie
end


#Retourne l'état de santé final de tous les composants avec le champ SoH 
function compute_aging(mg::Microgrid)

    return_dict = Dict()

    # Ici on itère sur un vecteur que l'on crée nous même, ce vecteur contient lui même 3 vecteurs, les storages, les converters et les generations
    for category in [mg.storages, mg.converters, mg.generations]
        # Chacun de ces vecteurs est maintenant désigné par la variable interne à la boucle for 'category'.
        # On conserve dans la "category" uniquement les éléments qui ont la propriété/ le champ SoH.
        # On réalise cela en gardant uniquement les indice qui ont valider le test 'hasproperty.(category, :soh)'
        for a in category[hasproperty.(category, :soh)]
            #Enfin on crée un champ dans le dictionnaire dont la clé est le type sous forme de chaine de caractère et la valeur est le SoH final.
            return_dict[string(typeof(a))] = a.soh[1,end,:]
        end
    end

    return return_dict
end

#Retourne le nombre de remplacement et l'état de santé consommé pour la batterie dans un dictionnaire 
function compute_aging(mg::Microgrid)

    # Chaque remplacement induit qu'au début de l'année, le SoH est à 1. La première année le SoH est aussi à 1 et on ne doit pas compoter cette occurence
    # le code 'sum(mg.storages[1].soh[1,:,:] .== 1) - 1' rempli ce role

    return_dict = Dict()

    for category in [mg.storages, mg.converters, mg.generations]
        
        for a in category[hasproperty.(category, :soh)]

            tmp_replacement = dropdims(sum(mg.storages[1].soh[1,:,:] .== 1, dims=1), dims = 1) .- 1

            tmp_consumed_health = 1. .- a.soh[1,end,:]

            tmp_tuple = (replacement = tmp_replacement, consumed_health = tmp_consumed_health)

            return_dict[string(typeof(a))] = tmp_tuple
        end
    end

    return return_dict
end


###################################################################


####################### Coût ######################################


function compute_cost(mg::Microgrid, designer::AbstractDesigner)
# On notera que avant de finaliser le calcul, on peut appliquer un taux d'actualisation 
    taux = 3/100
    γ = [1/(1+taux)^i for i in 0:(ny-1)]
# Il suffit alors de multiplier nos tableaux par γ

    #Calcul du cout d'opération
    opex = dropdims(sum(max.(0., mg.grids[1].carrier.power .* mg.grids[1].cost_in); dims = 1), dims=1)


    #Calcul du cout d'investissement, initial plus les remplacement
    capexx = zeros(ny,ns)
    for (component, init_decisions, repl_decisions)  in [(mg.storages, designer.storages, designer.decisions.storages), (mg.converters, designer.converters, designer.decisions.converters), (mg.generations, designer.generations, designer.decisions.generations)]
        for c in component
            #initial
            capexx[1,:] .+= c.cost[1,:] .* init_decisions[string(typeof(c))]
            #remplacement
            capexx .+= c.cost[:,:] .* repl_decisions[string(typeof(c))]
        end
    end


    return opex .+ capexx

end



###################################################################
#Il y'a plusieurs façons de renvoyer plusieurs valeurs depuis une fonction en julia, en voici 3 

#On fait un retour multiple où on peut iterer sur les valeur de retour 'a, b = f(c)'
function compute_metrics(mg::Microgrid, designer::AbstractDesigner)
    return compute_aging(mg), compute_cost(mg, designer), compute_RES3(mg)
end

#On fait une variable de retour groupée dans une structure, ici un namedTuple
function compute_metrics(mg::Microgrid, designer::AbstractDesigner)
    return (vieillissement = compute_aging(mg), cout = compute_cost(mg, designer), RES = compute_RES3(mg))
end

mutable struct AllMetrics
    vieillissement::Dict{Any,Any}
    cout::Matrix{Float64}
    RES::Matrix{Float64}
end

#On fait créer une structure adapté à la récéption des variables de retour.
function compute_metrics(mg::Microgrid, designer::AbstractDesigner)
    return AllMetrics(compute_aging(mg), compute_cost(mg, designer), compute_RES3(mg))
end
################################################################################








####################### Les plots ##################################



####################### PV production ##############################

hours = [i for i in 1:(8760*ny)]
y_values = zeros(nh*ny, ns)
labels = Matrix{String}(undef, 1, ns)

for s in 1:ns
    y_values[:,s] = vec(microgrid.generations[1].carrier.power[:,:,s])
    labels[s] = "scenario $s"
end

plot_pv = Plots.plot(hours, y_values, label = labels, linewidth=2, title = "PV production")
xlabel!("Hours")
ylabel!("Power (W)")
display(plot_pv)



####################### Load demand ################################

hours = [i for i in 1:(8760*ny)]
y_values = zeros(nh*ny, ns)
labels = Matrix{String}(undef, 1, ns)

for s in 1:ns
    y_values[:,s] = vec(microgrid.demands[1].carrier.power[:,:,s])
    labels[s] = "scenario $s"
end

plot_load = Plots.plot(hours, y_values, label = labels, linewidth=2, title="Load demand")
xlabel!("Hours")
ylabel!("Power (W)")
display(plot_load)

###################### Grouped #####################################


hours = [i for i in 1:(8760*ny)]
y_values_PV = zeros(nh*ny, ns)
y_values_load = zeros(nh*ny, ns)

labels = Matrix{String}(undef, 1, ns)

for s in 1:ns
    y_values_load[:,s] = vec(microgrid.demands[1].carrier.power[:,:,s])
    y_values_PV[:,s] = vec(microgrid.generations[1].carrier.power[:,:,s])
    labels[s] = "scenario $s"
end

Plots

plot_load = Plots.plot(hours, y_values_load, linewidth=2, title="Load demand", xlabel = "Hours",  legend=false)
ylabel!("Power (W)")
plot_pv = Plots.plot(hours, y_values_PV, linewidth=2, title = "PV production", legend=false)
ylabel!("Power (W)")
p = Plots.plot(plot_pv, plot_load, label = labels, layout = (2,1), legend=false)

display(p)


####################### SoC plot ###################################

# Attention le tableau soc va jusqu'a nh+1 et jusqu'a ny+1

hours = [i for i in 1:(8760*ny)]
y_values = zeros(nh*ny, ns)
labels = Matrix{String}(undef, 1, ns)


for s in 1:ns
    for a in microgrid.storages
        y_values[:,s] = vec(a.soc[1:end-1, 1:ny, s])
        labels[s] = string("$s : Storage : ", typeof(a))
    end
end

plot_soc = Plots.plot(hours, y_values, label = labels, linewidth=2, title="State-of-charge")
xlabel!("Hours")
ylabel!("SoC")
display(plot_soc)


####################### SoH plot ###################################

hours = [i for i in 1:(8760*ny)]
y_values = zeros(nh*ny, ns)
labels = Matrix{String}(undef, 1, ns)

for s in 1:ns
    for a in microgrid.storages
        y_values[:,s] = vec(a.soh[1:end-1, 1:ny, s])
        labels[s] = string("Storage : ", typeof(a))
    end
end

plot_soh = Plots.plot(hours, y_values, label = labels, linewidth=2, title="State-of-health")
xlabel!("Hours")
ylabel!("SoC")
display(plot_soh)


