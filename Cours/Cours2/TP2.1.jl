using JuMP

using JLD2, FileIO, Cbc, Ipopt #, Gurobi


using Gurobi



##################### Exemple ############################
m = Model(Gurobi.Optimizer)

@variable(m, x1, Int) # variable de décision 1
@variable(m, 4 >= x2 >= 0., Int) # variable de décision 2
n=12
@variable(m, affectation[1:n,1:n], Bin) # Une matrice de variable binaire n*n

@constraint(m, 2 * m[:x1] + 3 * m[:x2] <= 12 ) # Contrainte 1
@constraint(m, m[:x1] - m[:x2] >= -4 )# Contrainte 2
@constraint(m, m[:x1] >= 4 )

@constraint(m, [i in 1:n], sum(m[:affectation][i,j] for j in 1:n) >= 5 ) 
@constraint(m, [j in 1:n], sum(m[:affectation][i,j] for i in 1:n) <= 8 ) 


@objective(m, Max, m[:x1] * 10  +  m[:x2] * 16) # Fonction Objectif


JuMP.optimize!(m) # résout le modèle de façon optimale
 
objective_value(m) # Retourne la valeur de la fonction objectif

value(m[:x1]) #Intérroge la valeur prise par la variable x1
value(m[:x2])

value.(m[:affectation])



#################### Exercice 1 #######################
##################### Données ##########################


marge_C = 700 # marge vélo cargo pour 100 vélos
marge_S = 300 # marge vélo standard pour 100 vélos

lim_C = 7 # limite de fabrication pour 100 vélos

h_C = 6 # temps pour 100 vélos cargos
h_S = 5 # temps pour 100 vélos standards
h_lim = 60 # Limite de temps totale

s_C = 250 # surface pour 100 vélos cargos
s_S = 100 # surface pour 100 vélos
s_lim = 1500 # limite de surface

m = Model(Cbc.Optimizer)



#################### Exercice 2 #######################
##################### Données ##########################
matrice_cout = [3.0  5.0  6.0  4.0  10.0
                6.0  8.0  3.0  2.0   1.0
                8.0  9.0  0.0  1.0   2.0
                3.0  9.0  5.0  6.0   1.0
                10.0  3.0  4.0  2.0   7.0]

n = 5 #Nombre d'employé (implique une matrice n*n)                

                
matrice_cout_n = round.(rand(n,n) .* 10) # génère une matrice de cout n*n allant de 0 à 10

















##################### Solutions Ex1 ##########################
m = Model(Cbc.Optimizer)

#variables de décisions
@variable(m, C >= 0., Int) # Le nombre de lot de 100 vélos cargos
@variable(m, S >= 0., Int) # Le nombre de lot de 100 vélos standards

@constraint(m, h_C * m[:C] + h_S * m[:S] <= h_lim ) # Contrainte de travail
@constraint(m, s_C * m[:C] + s_S * m[:S] <= s_lim )# Contrainte de surface
@constraint(m, m[:C] <= lim_C ) # Contrainte sur le nombre de vélo cargo

@objective(m, Max, m[:C] * marge_C  +  m[:S] * marge_S) # Fonction Objectif


JuMP.optimize!(m) # résout le modèle de façon optimale
 
objective_value(m) # Retourne la valeur de la fonction objectif


value(m[:C]) #Intérroge la valeur prise par la variable C
value(m[:S])





##################### Solutions Ex2 ##########################
m = Model(Cbc.Optimizer)

#variables de décisions
@variable(m, affectation[1:n,1:n], Bin) # Une matrice de variable binaire n*n

@constraint(m, [t in 1:n], sum(m[:affectation][p,t] for p in 1:n) == 1 ) # chaque personne à une seule tâche 
@constraint(m, [p in 1:n], sum(m[:affectation][p,t] for t in 1:n) == 1 ) # chaque tâche est réalisée par 1 seule personne

@objective(m, Max, sum(m[:affectation][p,t] * matrice_cout[p,t] for p in 1:n for t in 1:n)) # Fonction Objectif


JuMP.optimize!(m) # résout le modèle de façon optimale
 
objective_value(m) # Retourne la valeur de la fonction objectif

value.(m[:affectation]) #Retourne la matrice de décision








################ Données Exercice 3 #####################
# 1 scénario de 1 année avec 8760 pas de temps, donc un pas horaire.

include(joinpath(pwd(),"src","Genesys2.jl"))

nh, ny, ns = 8760, 1, 1

h_interval = 1:nh

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Le microgrid étudié
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion(eff_model = FixedLiionEfficiency(), SoH_model = FixedLifetimeLiion(), couplage = (E=false, R = false)),
                Grid(carrier = Electricity()))
                

using JLD2, FileIO

# Récupération des variables d'environement

data_fix = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4.jld2"))
data_HP_HC = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4_HP_HC.jld2"))

data_selected = data_fix
     
ω_a = Scenarios(microgrid, data_selected, true; seed=1:ns)



h_interval = 1:720

#################### Fonction pour créer le modèle ########################
function get_model_1(solver, microgrid, ω_a)

    # Couts associés au grid 
    cost_in = ω_a.grids[1].cost_in[h_interval,1,1] #Prix d'achat €/kWh
    cost_out = ω_a.grids[1].cost_out[h_interval,1,1] #Prix de vente €/kWh
    cout_depassement = microgrid.grids[1].cost_exceed[1,1] # Cout de dépassement de la puissance souscrite au réseau €/h

    # Variables d'environement (imposées)
    p_load = ω_a.demands[1].power[h_interval,1,1] # Demande en kWh
    p_gen = ω_a.generations[1].power[h_interval,1,1] # Puissance par unité de kWc installé 

    #### Configuration des variables décrivant les composant pour le modèle de prog mathématique
    liion = microgrid.storages[1]

    η_self = liion.eff_model.η_self #Facreur d'auto-décharge
    η = liion.eff_model.η_ch # Rendement
    ∆h = 1. # taille du pas d'opération
    seuil_max = liion.α_soc_max # SoC max
    seuil_min = liion.α_soc_min # SoC min
    C_rate = liion.eff_model.α_p_ch # C-rate max

    Erated = 20. # Capacité de la batterie kWh
    PV = 10. # Puissance du panneau solaire kWc
    grid_seuil = 10. # puissance souscrite au réseau kW

    #Déclaration du model et du solver
    m1 = Model(solver.Optimizer)
#################################################
    # insérez vos équations ici
    # Variables
    # Contraintes
    # Objectif
#################################################

    # Utilisez tout ou partie des données ci-dessus
    return m1

end



################## Execution du modèle #############################
mod1 = get_model_1(Cbc, microgrid, ω_a)
JuMP.optimize!(mod1)

println("La solution optimale vaut : ", round(objective_value(mod1), digits=2), " €")
println("Le problème à été résolu en : ", round(solve_time(mod1), digits=2), " secondes")





#################### Exercice 3 Solutions #######################


################################################################
######################### Model 1 ##############################
################################################################

function get_model_1(h_interval)

    # Couts associés au grid 
    cost_in = ω_a.grids[1].cost_in[h_interval,1,1] #Prix d'achat €/kWh
    cost_out = ω_a.grids[1].cost_out[h_interval,1,1] #Prix de vente €/kWh
    cout_depassement = microgrid.grids[1].cost_exceed[1,1] # Cout de dépassement de la puissance souscrite au réseau €/h

    # Variables d'environement (imposées)
    p_load = ω_a.demands[1].power[h_interval,1,1] # Demande en kWh
    p_gen = ω_a.generations[1].power[h_interval,1,1] # Puissance par unité de kWc installé 

    #### Configuration des variables décrivant les composant pour le modèle de prog mathématique
    liion = microgrid.storages[1]

    η_self = liion.eff_model.η_self #Facreur d'auto-décharge
    η = liion.eff_model.η_ch # Rendement
    ∆h = 1. # taille du pas d'opération
    seuil_max = liion.α_soc_max # SoC max
    seuil_min = liion.α_soc_min # SoC min
    C_rate = liion.eff_model.α_p_ch # C-rate max

    Erated = 20. # Capacité de la batterie kWh
    PV = 10. # Puissance du panneau solaire kWc
    grid_seuil = 10. # puissance souscrite au réseau kW

    #Déclaration du model et du solver
    m1 = Model(Gurobi.Optimizer)

    #variables de décisions
    @variable(m1, p_ch[1:length(h_interval)] >= 0.)
    @variable(m1, p_dch[1:length(h_interval)] >= 0.)

    #variables de recours (formulée comme des variables de décisions)
    @variable(m1, p_in[1:length(h_interval)] >= 0.)
    @variable(m1, p_out[1:length(h_interval)] >= 0.)

    # variables d'état, formulé comme un variable de décisions mais contraintes.
    @variable(m1, soc[1:(length(h_interval)+1)])

    @constraints(m1, begin
    #dynamique de l'état de la batterie
        [h in 1:length(h_interval)], m1[:soc][h+1] == m1[:soc][h] * (1-η_self * ∆h) - (m1[:p_dch][h] / η - m1[:p_ch][h] * η) * ∆h
    #bornes du soc
        [h in 1:(length(h_interval)+1)], m1[:soc][h] <= seuil_max * Erated
        [h in 1:(length(h_interval)+1)], m1[:soc][h] >= seuil_min * Erated
    #Borne de puissance
        [h in 1:length(h_interval)], m1[:p_ch][h] <= Erated * C_rate
        [h in 1:length(h_interval)], m1[:p_dch][h] <= Erated * C_rate
    #initialisation et périodicité
        m1[:soc][1] == 0.5 * Erated
        m1[:soc][end] >= m1[:soc][1]
    end)

            
    @constraints(m1, begin
        [h in 1:length(h_interval)],  p_load[h] - (p_gen[h] * PV) + m1[:p_ch][h] - m1[:p_dch][h] - m1[:p_in][h] + m1[:p_out][h] == 0
    end)

    @objective(m1, Min, sum((m1[:p_in][h] * cost_in[h] .- m1[:p_out][h] .* cost_out[h]) for h in 1:length(h_interval)) )

    return m1

end
######################################################################################
######################################################################################
######################################################################################



################################################################
######################### Model 2 ##############################
################################################################

function get_model_2(h_interval)

    # Couts associés au grid 
    cost_in = ω_a.grids[1].cost_in[h_interval,1,1] #Prix d'achat €/kWh
    cost_out = ω_a.grids[1].cost_out[h_interval,1,1] #Prix de vente €/kWh
    cout_depassement = microgrid.grids[1].cost_exceed[1,1] # Cout de dépassement de la puissance souscrite au réseau €/h

    # Variables d'environement (imposées)
    p_load = ω_a.demands[1].power[h_interval,1,1] # Demande en kWh
    p_gen = ω_a.generations[1].power[h_interval,1,1] # Puissance par unité de kWc installé 

    #### Configuration des variables décrivant les composant pour le modèle de prog mathématique
    liion = microgrid.storages[1]

    η_self = liion.eff_model.η_self #Facreur d'auto-décharge
    η = liion.eff_model.η_ch # Rendement
    ∆h = 1. # taille du pas d'opération
    seuil_max = liion.α_soc_max # SoC max
    seuil_min = liion.α_soc_min # SoC min
    C_rate = liion.eff_model.α_p_ch # C-rate max

    Erated = 20. # Capacité de la batterie kWh
    PV = 10. # Puissance du panneau solaire kWc
    grid_seuil = 10. # puissance souscrite au réseau kW

    # On ajoute à la fonction de cout une notion de dépassement. On considère un seuil de demande horaire au dessus duquel on paye un surcout
   
    M = 10000. #Valeur big-M pour les contraintes de type "if-else"

    #Déclaration du model et du solver
    m2 = Model(Gurobi.Optimizer)

    #variables de décisions
    @variable(m2, p_ch1[1:length(h_interval)] >= 0.)
    @variable(m2, p_ch2[1:length(h_interval)] >= 0.)

    @variable(m2, p_dch1[1:length(h_interval)] >= 0.)
    @variable(m2, p_dch2[1:length(h_interval)] >= 0.)


    @variable(m2, z1[1:(length(h_interval))], Bin)
    @variable(m2, z2[1:(length(h_interval))], Bin)
    eff1 = 0.97
    eff2 = 0.93


    #variables de recours (formulée comme des variables de décisions)
    @variable(m2, p_in[1:length(h_interval)] >= 0.)
    @variable(m2, p_out[1:length(h_interval)] >= 0.)

    # variables d'état, formulé comme un variable de décisions mais contraintes.
    @variable(m2, soc[1:(length(h_interval)+1)])

    #La variable de dépassement
    @variable(m2, depassement[1:(length(h_interval))], Bin)

    @constraints(m2, begin
        #Toujours un unique morceau actif
        [h in 1:length(h_interval)], m2[:z1][h] + m2[:z2][h] == 1

        # Le rendement 1 c'est pour les valeur de C_rate 1/2 de max
        [h in 1:length(h_interval)], m2[:p_dch1][h] <= m2[:z1][h] * C_rate * Erated /2
        [h in 1:length(h_interval)], m2[:p_ch1][h] <= m2[:z1][h] * C_rate * Erated /2

        [h in 1:length(h_interval)], m2[:p_dch2][h] <= m2[:z2][h] * C_rate * Erated
        [h in 1:length(h_interval)], m2[:p_ch2][h] <= m2[:z2][h] * C_rate * Erated

    end)


    @constraints(m2, begin
    #dynamique de l'état de la batterie
        [h in 1:length(h_interval)], m2[:soc][h+1] == m2[:soc][h] * (1-η_self * ∆h) - (m2[:p_dch1][h] / eff1 + m2[:p_dch2][h] / eff2 - m2[:p_ch1][h] * eff1 - m2[:p_ch2][h] * eff2) * ∆h
    #bornes du soc
        [h in 1:(length(h_interval)+1)], m2[:soc][h] <= seuil_max * Erated
        [h in 1:(length(h_interval)+1)], m2[:soc][h] >= seuil_min * Erated

    #initialisation et périodicité
        m2[:soc][1] == 0.5 * Erated
        m2[:soc][end] >= m2[:soc][1]
    end)

            
    @constraints(m2, begin
        [h in 1:length(h_interval)],  p_load[h] - (p_gen[h] * PV) + m2[:p_ch1][h] + m2[:p_ch2][h] - m2[:p_dch1][h] - m2[:p_dch2][h] - m2[:p_in][h] + m2[:p_out][h] == 0
    end)

    @constraints(m2, begin
        [h in 1:length(h_interval)],  m2[:depassement][h] * M >= m2[:p_in][h] - grid_seuil #  p_in > grid_seuil => depassement = 1
        [h in 1:length(h_interval)],  (m2[:depassement][h]-1) * M <= m2[:p_in][h] - grid_seuil # p_in < grid_seuil => depassement = 0
    end)

    @objective(m2, Min, sum((m2[:p_in][h] * cost_in[h] .- m2[:p_out][h] .* cost_out[h] .+ m2[:depassement][h] * cout_depassement) for h in 1:length(h_interval)) )

    return m2

end
######################################################################################
######################################################################################
######################################################################################



################################################################
######################### Model 3 ##############################
################################################################

function get_model_3(h_interval)

     # Couts associés au grid 
     cost_in = ω_a.grids[1].cost_in[h_interval,1,1] #Prix d'achat €/kWh
     cost_out = ω_a.grids[1].cost_out[h_interval,1,1] #Prix de vente €/kWh
     cout_depassement = microgrid.grids[1].cost_exceed[1,1] # Cout de dépassement de la puissance souscrite au réseau €/h
 
     # Variables d'environement (imposées)
     p_load = ω_a.demands[1].power[h_interval,1,1] # Demande en kWh
     p_gen = ω_a.generations[1].power[h_interval,1,1] # Puissance par unité de kWc installé 
 
     #### Configuration des variables décrivant les composant pour le modèle de prog mathématique
     liion = microgrid.storages[1]
 
     η_self = liion.eff_model.η_self #Facreur d'auto-décharge
     η = liion.eff_model.η_ch # Rendement
     ∆h = 1. # taille du pas d'opération
     seuil_max = liion.α_soc_max # SoC max
     seuil_min = liion.α_soc_min # SoC min
     C_rate = liion.eff_model.α_p_ch # C-rate max
 
     Erted = 20. # Capacité de la batterie kWh
     PV = 10. # Puissance du panneau solaire kWc
     grid_seuil = 10. # puissance souscrite au réseau kW
        
    # On ajoute à la fonction de cout une notion de dépassement. On considère un seuil de demande horaire au dessus duquel on paye un surcout
    M = 10000. #Valeur big-M pour les contraintes de type "if-else"

    #Déclaration du model et du solver
    m3 = Model(Gurobi.Optimizer)
    set_attribute(m3, "non_convex", 2)
    set_optimizer_attribute(m3, "MIPGap", 10^(-2))


    #variables de décisions
    @variable(m3, p_ch[1:length(h_interval)] >= 0.)
    @variable(m3, p_dch[1:length(h_interval)] >= 0.)

    #variables de recours (formulée comme des variables de décisions)
    @variable(m3, p_in[1:length(h_interval)] >= 0.)
    @variable(m3, p_out[1:length(h_interval)] >= 0.)

    @variable(m3, η_ch[1:length(h_interval)] >= 0.)
    @variable(m3, η_dch[1:length(h_interval)] >= 0.)

    # variables d'état, formulé comme un variable de décisions mais contraintes.
    @variable(m3, soc[1:(length(h_interval)+1)])

    #La variable de dépassement
    @variable(m3, depassement[1:(length(h_interval))], Bin)


    @variable(m3, Erated[1:(length(h_interval)+1)] >= 0)    


    @constraints(m3, begin
    #dynamique de l'état de la batterie
    [h in 1:length(h_interval)], m3[:soc][h+1] == m3[:soc][h] * (1-η_self * ∆h) - (m3[:p_dch][h] * m3[:η_dch][h] - m3[:p_ch][h] * m3[:η_ch][h]) * ∆h
    #dynamique du soh 
    [h in 1:length(h_interval)], m3[:Erated][h+1] == m3[:Erated][h] - (m3[:p_dch][h] * m3[:η_dch][h] + m3[:p_ch][h] * m3[:η_ch][h]) * 0.002
    #bornes du soc
        [h in 1:(length(h_interval)+1)], m3[:soc][h] <= seuil_max * m3[:Erated][h]

        [h in 1:(length(h_interval)+1)], m3[:soc][h] >= seuil_min * m3[:Erated][h]

    #Borne de puissance
        [h in 1:length(h_interval)], m3[:p_ch][h] <=  m3[:Erated][h] * C_rate
        [h in 1:length(h_interval)], m3[:p_dch][h] <=  m3[:Erated][h] * C_rate
    #initialisation et périodicité
        m3[:soc][1] == 0.5 * m3[:Erated][1]
        m3[:soc][end] >= m3[:soc][1]

        m3[:Erated][1] == Erted

        [h in 1:length(h_interval)], m3[:η_dch][h] == 2.05 - m3[:Erated][h]/Erted  
        [h in 1:length(h_interval)], m3[:η_ch][h] == m3[:Erated][h]/Erted - 0.05

    end)

    @constraints(m3, begin
        [h in 1:length(h_interval)],  p_load[h] - (p_gen[h] * PV) + m3[:p_ch][h] - m3[:p_dch][h] - m3[:p_in][h] + m3[:p_out][h] == 0
    end)

    @constraints(m3, begin
        [h in 1:length(h_interval)],  m3[:depassement][h] * M >= m3[:p_in][h] - grid_seuil #  p_in > grid_seuil => depassement = 1
        [h in 1:length(h_interval)],  (m3[:depassement][h]-1) * M <= m3[:p_in][h] - grid_seuil # p_in < grid_seuil => depassement = 0
    end)

    @objective(m3, Min, sum((m3[:p_in][h] * cost_in[h] .- m3[:p_out][h] .* cost_out[h] .+ m3[:depassement][h] * cout_depassement) for h in 1:length(h_interval)) )

    return m3
end
######################################################################################
######################################################################################
######################################################################################


h_interval = 1:144

mod1 = get_model_1(h_interval)
JuMP.optimize!(mod1)

println("La solution optimale vaut : ", round(objective_value(mod1), digits=2), " €")
println("Le problème à été résolu en : ", round(solve_time(mod1), digits=2), " secondes")




mod2 = get_model_2(h_interval)
JuMP.optimize!(mod2)

println("La solution optimale vaut : ", round(objective_value(mod2), digits=2), " €")
println("Le problème à été résolu en : ", round(solve_time(mod2), digits=2), " secondes")



mod3 = get_model_3(h_interval)
JuMP.optimize!(mod3)

println("La solution optimale vaut : ", round(objective_value(mod3), digits=2), " €")
println("Le problème à été résolu en : ", round(solve_time(mod3), digits=2), " secondes")

d_max= 10
T1 = zeros(d_max)
T2 = zeros(d_max)
T3 = zeros(d_max)

for i in 1:d_max
    h_interval = 1:(i*24)

    mod1 = get_model_1(h_interval)
    JuMP.optimize!(mod1)
    
    println("La solution optimale vaut : ", round(objective_value(mod1), digits=2), " €")
    println("Le problème à été résolu en : ", round(solve_time(mod1), digits=2), " secondes")

    T1[i] = solve_time(mod1)
    
    mod2 = get_model_2(h_interval)
    JuMP.optimize!(mod2)
    
    println("La solution optimale vaut : ", round(objective_value(mod2), digits=2), " €")
    println("Le problème à été résolu en : ", round(solve_time(mod2), digits=2), " secondes")
    
    T2[i] = solve_time(mod2)

    mod3 = get_model_3(h_interval)
    JuMP.optimize!(mod3)
    
    println("La solution optimale vaut : ", round(objective_value(mod3), digits=2), " €")
    println("Le problème à été résolu en : ", round(solve_time(mod3), digits=2), " secondes")

    T3[i] = solve_time(mod3)


end



days = [x for x in 1:365]

d = Dict("LP"=>T1, "MILP"=>T2, "MINLP"=>T3, "days"=> days)
CSV.write("duration.csv", d)









##############################################################################################
######################## Comparaison des Méthodes de controle ################################
##############################################################################################
include(joinpath(pwd(),"src","Genesys2.jl"))


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

##################################################################
#Executez à nouveau vos RB
####################################################################:

# 1 scénario de 1 année avec 8760 pas de temps, donc un pas horaire.
nh, ny, ns = 8760, 1, 1

h_interval = 1:nh

###### Création du microréseau (on y met les modèles les plus simples car ce sont ceux que l'on modélise avec la prog mathématique)
microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion(eff_model = FixedLiionEfficiency(), SoH_model = SemiEmpiricalLiion(), couplage = (E=false, R = false)),
                Grid(carrier = Electricity()))
                

using JLD2, FileIO

# Récupération des variables d'environement

data_fix = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4.jld2"))
data_HP_HC = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4_HP_HC.jld2"))

data_selected = data_fix
     
ω_a = Scenarios(microgrid, data_selected, true; seed=1:ns)

cost_in = ω_a.grids[1].cost_in[h_interval,1,1]
cost_out = ω_a.grids[1].cost_out[h_interval,1,1]
cout_depassement = microgrid.grids[1].cost_exceed[1,1]

p_load = ω_a.demands[1].power[h_interval,1,1]
p_gen = ω_a.generations[1].power[h_interval,1,1]

#### Configuration des variables décrivant les composant pour le modèle de prog mathématique

liion = microgrid.storages[1]

η_self = liion.eff_model.η_self
η = liion.eff_model.η_ch
∆h = 1.
seuil_max = liion.α_soc_max
seuil_min = liion.α_soc_min
C_rate = liion.eff_model.α_p_ch


Erated = 20.
PV = 10.
grid_seuil = 10.

generations = Dict("Solar" => 10.)
storages = Dict("Liion" => 20.)
subscribed_power = Dict("Electricity" => 10.)

###################################################################

############## Evaluation ###############
designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω_a)
controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = 101)), ω_a)

simulate!(microgrid, controller, designer, ω_a, options = Options(mode = "serial"))

metrics = Metrics(microgrid, designer)

# Cout pour RB
metrics.cost.opex 

# Cout pour le model math opt
m2 = model1(Cbc)
JuMP.optimize!(m2)
objective_value(m2)

# Affichage des décisions de batterie
RB_p_bat = microgrid.storages[1].carrier.power


balance = p_load - (p_gen * PV) + value.(m2[:p_ch]) -  value.(m2[:p_dch]) -  value.(m2[:p_in]) +  value.(m2[:p_out])


p_net = p_load - (p_gen * PV) 
p_bat_mathopt = value.(m2[:p_ch]) -  value.(m2[:p_dch])
p_grid_mathopt = - value.(m2[:p_in]) +  value.(m2[:p_out])

balance = p_net + p_bat_mathopt + p_grid_mathopt


figure("RB")
PyPlot.plot(1:8760, RB_p_bat[:,1,1], label = "p_bat")
PyPlot.plot(1:8760, p_net, label = "p_net")
PyPlot.plot(1:8760, microgrid.grids[1].carrier.power[:,1,1], label = "grid")
legend()

figure("Math opt")
PyPlot.plot(1:8760, -p_bat_mathopt , label = "p_bat" )
PyPlot.plot(1:8760, p_net, label = "p_net"  )
PyPlot.plot(1:8760,  -p_grid_mathopt , label = "grid"  )
legend()



##############################################################################################
##################### Mise à jour pour faire un model de design/contrôle #####################
##############################################################################################


function model_design1(solver, Erated_price, PV_price)

    #Déclaration du model et du solver
    md1 = Model(solver.Optimizer)

    #Design variables
    @variable(md1, Erated >= 0.)
    @variable(md1, PV >= 0.)

    @constraint(md1, md1[:Erated] <= max_Erated)
    @constraint(md1, md1[:PV] <= max_PV)

    #variables de décisions
    @variable(md1, p_ch[1:length(h_interval)] >= 0.)
    @variable(md1, p_dch[1:length(h_interval)] >= 0.)

    #variables de recours (formulée comme des variables de décisions)
    @variable(md1, p_in[1:length(h_interval)] >= 0.)
    @variable(md1, p_out[1:length(h_interval)] >= 0.)

    # variables d'état, formulé comme un variable de décisions mais contraintes.
    @variable(md1, soc[1:(length(h_interval)+1)])

    @constraints(md1, begin
    #dynamique de l'état de la batterie
        [h in 1:length(h_interval)], md1[:soc][h+1] == md1[:soc][h] * (1-η_self * ∆h) - (md1[:p_dch][h] / η - md1[:p_ch][h] * η) * ∆h
    #bornes du soc
        [h in 1:(length(h_interval)+1)], md1[:soc][h] <= seuil_max * md1[:Erated]
        [h in 1:(length(h_interval)+1)], md1[:soc][h] >= seuil_min * md1[:Erated]
    #Borne de puissance
        [h in 1:length(h_interval)], md1[:p_ch][h] <= md1[:Erated] * C_rate
        [h in 1:length(h_interval)], md1[:p_dch][h] <= md1[:Erated] * C_rate
    #initialisation et périodicité
        md1[:soc][1] == 0.5 * md1[:Erated]
        md1[:soc][end] >= md1[:soc][1]
    end)

            
    @constraints(md1, begin
        [h in 1:length(h_interval)],  p_load[h] - (p_gen[h] * md1[:PV]) + md1[:p_ch][h] - md1[:p_dch][h] - md1[:p_in][h] + md1[:p_out][h] == 0
    end)

    @expression(md1, opex, sum((md1[:p_in][h] * cost_in[h] .- md1[:p_out][h] .* cost_out[h]) for h in 1:length(h_interval)) )
    @expression(md1, capex, md1[:Erated] * Erated_price + md1[:PV] * PV_price)


    @objective(md1, Min, opex + capex )

    return md1

end


Erated_price = 300  #€/kwh
PV_price = 1300 #€/kWp

j = 1  #j-1 < 365 - njours
nJours = 365 
h_interval = ((j-1)*24+1):(24*(j+nJours-1))

data = data_fix

# Chargement des variables d'environement
p_gen = data["pv"].power[h_interval,1,1]
p_load = data["ld_E"].power[h_interval,1,1]
cost_in = data["grid_Elec"].cost_in[h_interval,1,1]
cost_out = data["grid_Elec"].cost_out[h_interval,1,1]


η_self = 0.001
η = 0.95
∆h = 1.
seuil_max = 0.9
seuil_min = 0.1
C_rate = 1.

max_Erated = 100.
max_PV = 100.


mdesign = model_design1(Cbc, Erated_price, PV_price)
JuMP.optimize!(mdesign)
objective_value(mdesign)

value(mdesign[:Erated])
value(mdesign[:PV])





















#################### Génération de résultat en faisant varier les prix #########################
# Définition de la fenêtre temporelle


j = 1  #j < 365 - njours
nJours = 365 
h_interval = ((j-1)*24+1):(24*(j+nJours-1))

# Chargement des variables d'environement
p_gen = data["pv"].power[h_interval,1,1]
p_load = data["ld_E"].power[h_interval,1,1]
cost_in = data["grid_Elec"].cost_in[h_interval,1,1]
cost_out = data["grid_Elec"].cost_out[h_interval,1,1]


η_self = 0.001
η = 0.95
∆h = 1.
seuil_max = 0.9
seuil_min = 0.1
C_rate = 1.

max_Erated = 100.
max_PV = 100.



Erated_prices = [i for i in 500:-30:10]
PV_prices = [i for i in 1510:-100:10]

pv_installed = zeros(length(Erated_prices), length(PV_prices))
bat_installed = zeros(length(Erated_prices), length(PV_prices))

for (i, Erated_price) in enumerate(Erated_prices)
    for (j, PV_price) in enumerate(PV_prices)

        m = model_design1(Gurobi, Erated_price, PV_price)
        JuMP.optimize!(m)

        bat_installed[i,j] = value(m[:Erated])
        pv_installed[i,j] = value(m[:PV])

    end
end



using PyCall
np = pyimport("numpy")


xticks = np.linspace(1, length(PV_prices) , 16, dtype=np.int)
xtickslabs = [PV_prices[idx] for idx in xticks]
 
yticks= np.linspace(1, length(Erated_prices) , 17, dtype=np.int)
yticklabs = [Erated_prices[idx] for idx in yticks]

subplot(1, 2, 1)
ax = Seaborn.heatmap(pv_installed, linewidth=0.5, xticklabels=xtickslabs, yticklabels=yticklabs, cmap="jet")
ax.set_title("Installed PV", fontsize = 24)
plt.xlabel("Battery price [€/kWh]", fontsize = 20)
plt.ylabel("PV price [€/kWc]",  fontsize = 20)
ax.set_xticklabels(xtickslabs, fontsize = 18)
ax.set_yticklabels(yticklabs, fontsize = 18)
ax.collections[1].colorbar.ax.tick_params(labelsize=18)

subplot(1, 2, 2)
ax2 = Seaborn.heatmap(bat_installed, linewidth=0.5, xticklabels=xtickslabs, yticklabels=yticklabs, cmap="jet")
ax2.set_title("Installed Battery", fontsize = 24)
plt.xlabel("Battery price [€/kWh]",  fontsize = 20)
plt.ylabel("PV price [€/kWc]",  fontsize = 20)
ax2.set_xticklabels(xtickslabs, fontsize = 18)
ax2.set_yticklabels(yticklabs, fontsize = 18)
ax2.collections[1].colorbar.ax.tick_params(labelsize=18)







################# Comparaison temps de calcul #####################

# On récupère des données 
data_fix = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4.jld2"))
data_HP_HC = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4_HP_HC.jld2"))

# On séléctionne celles à utiliser
data = data_fix


using Seaborn

pygui(true)
figure("Environement Data (net power)")
Seaborn.plot(h_interval, p_load - (p_gen * PV))
Seaborn.plot(h_interval, repeat([0], length(h_interval)), c="black")

interval_j = [i for i in 1:30]

times = zeros(length(interval_j), 3)
obj = zeros(length(interval_j), 3)


for (i,nJours) in enumerate(interval_j)
    # Définition de la fenêtre temporelle
    j = 2  #j < 365 - njours
    h_interval = ((j-1)*24+1):(24*(j+nJours-1))

    # Chargement des variables d'environement
    p_gen = data["pv"].power[h_interval,1,1]
    p_load = data["ld_E"].power[h_interval,1,1]
    cost_in = data["grid_Elec"].cost_in[h_interval,1,1]
    cost_out = data["grid_Elec"].cost_out[h_interval,1,1]

    η_self = 0.001
    η = 0.95
    ∆h = 1.
    Erated = 20.
    PV = 10.
    seuil_max = 0.9
    seuil_min = 0.1
    C_rate = 1.

    grid_seuil = 6.
    cout_depassement = 0.01

    m1 = get_model_1(Gurobi)
    m2 = get_model_2(Gurobi)
    m3 = get_model_3()


    JuMP.optimize!(m1)
    JuMP.optimize!(m2)
    JuMP.optimize!(m3)

    obj[i,1] = objective_value(m1)
    obj[i,2] = objective_value(m2)
    obj[i,3] = objective_value(m3)

    times[i,1] = solve_time(m1)
    times[i,2] = solve_time(m2)
    times[i,3] = solve_time(m3)

end

ticksize = 20
legendsize = 28
labelsize = 28
titlesize = 24

figure("Comparaison de l'évolution du temps en fonction du nombre de jours")
fig, ax = plt.subplots()
for i in 1:3
    n=3
    vs = times[:,i]
    serie = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
    Seaborn.plot(interval_j[2:(end-1)], serie, label=string("model", i), linewidth=3)
end

plt.xlabel("nDays", fontsize=labelsize)
plt.ylabel("Solving time", fontsize=labelsize)
legend(fontsize=legendsize)
plt.yticks(fontsize=ticksize)
plt.xticks(fontsize=ticksize)


ax2 = PyPlot.axes([0.4, 0.55, .3, .25])
for i in 1:3
    n=3
    vs = times[:,i]
    serie = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]
    Seaborn.plot(interval_j[2:(end-1)], serie, label=string("model", i), linewidth=2)
end
ax2.set_title("zoom", fontsize=titlesize)
plt.yticks(fontsize=ticksize)
plt.xticks(fontsize=ticksize)
ax2.set_ylim([0.,maximum(times[:,2])])


