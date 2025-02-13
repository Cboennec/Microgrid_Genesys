
######################## TP 2.2 Métaheuristique ################################
include(joinpath(pwd(),"src","Genesys2.jl"))

nh, ny, ns = 8760, 2, 2

plotlyjs() 

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion(eff_model = PolynomialLiionEfficiency(), SoH_model = SemiEmpiricalLiion()),
                Grid(carrier = Electricity()))
                
using JLD2, FileIO

data_fix = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4.jld2"))
data_HP_HC = JLD2.load(joinpath(pwd(), "Cours", "Cours1", "data_light_4_HP_HC.jld2"))
     
ω_opti = Scenarios(microgrid, data_HP_HC, true; seed=1:ns)
ω_eval = Scenarios(microgrid, data_HP_HC, true; seed=(ns+1):(2ns))

#Métaheuristique avec RB du TP1

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


function RB_autonomie(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::RBC)
    controller.decisions.storages[1][h,y,s] = mg.demands[1].carrier.power[h,y,s] - mg.generations[1].carrier.power[h,y,s]
end



#################################################################
##################### Modification de la fonction obj ###########
#################################################################


# Objective functions
function fobj_cours(decisions::Array{Float64,1}, mg::Microgrid, designer::Metaheuristic, ω::Scenarios, varID::Dict)
    # Paramters
    nh, ny, ns = size(ω.demands[1].power)

    # Initialize mg
    mg_m = deepcopy(mg)
    mg_m.parameters = GlobalParameters(nh, ny, ns)

    # Initialize with the manual designer
    converters, generations, storages, subscribed_power = Dict(), Dict(), Dict(), Dict()
    for a in mg_m.converters
        if a isa Electrolyzer
            converters[string(typeof(a))] = (surface = decisions[varID[string(typeof(a))].surface], N_cell=Int64(round(decisions[varID[string(typeof(a))].N_cell])))
        elseif a isa FuelCell 
            converters[string(typeof(a))] = (surface = decisions[varID[string(typeof(a))].surface], N_cell=Int64(round(decisions[varID[string(typeof(a))].N_cell])))
        elseif a isa Heater
            converters[string(typeof(a))] = decisions[varID[string(typeof(a))]]
        end
    end
    for a in mg_m.generations
        generations[string(typeof(a))] = decisions[varID[string(typeof(a))]]
    end
    for a in mg_m.storages
        storages[string(typeof(a))] = decisions[varID[string(typeof(a))]]  
    end
    for a in mg_m.grids
        subscribed_power[string(typeof(a.carrier))] = decisions[varID[string(typeof(a.carrier))]] 
    end
    

    designer_m = initialize_designer!(mg_m, Manual(generations = generations, storages = storages, converters = converters, subscribed_power = subscribed_power), ω)


    ######################## Solution #######################################
    # Initialize controller
    controller_m = initialize_controller!(mg_m, designer.options.controller, ω)

    # Simulate
    simulate!(mg_m, controller_m, designer_m, ω, options = Options(mode = "serial"))

    # Metrics
    metrics = Metrics(mg_m, designer_m)


    return mean(metrics.npv.total) 

    ################################################################
end




varID = Dict("Solar" => 1, "Liion" => 2, "Electricity" => 3)

ub_var = [200., 1000., 50.]
lb_var = [1., 1., 1.] 

controller = RBC(options = RBCOptions(policy_selection = 101)) # Executez vos RB du TP1 ou celles de la corrections pour avoir accès à ces 



designer = initialize_designer!(microgrid, 
Metaheuristic(options = MetaheuristicOptions(;method = Clearing(nind = 50), multithreads=false, iterations = 20, controller = controller)),
 ω_opti,
 ub_var,
 lb_var,
 varID;
 f_obj = fobj_cours)  


 controller_eval = initialize_controller!(microgrid, controller, ω_opti)

  # Simulate 
  # Si vous utilisez le(s) scénario(s) de l'optim (ω_optim) vous retrouverez les même metrics.
  # Si vous utilisez des scénarios différents (ω_eval) pour l'évaluation vous obtiendrez des métriques différentes.
  simulate!(microgrid, controller_eval, designer, ω_eval, options = Options(mode = "serial"))

  # metrcis
  metrics = Metrics(microgrid, designer)







# On remarque qu'avec seulement 3 variables, l'optimisation est réalisée sans problème, avec une covnergence rapide
# On remarque aussi que la RB influence beaucoup le dimentionnement de la batterie mais pas vraiment celui du PV
# Note : Les résultats peuvent légèrement varier d'une execution  à l'autre car c'est un processus stochastique.


# la fobj doit répondre à un cahier des charge clair en terme d'input et d'output. 



















 #################### Solution possible #################
            # à insérer dans la fonction objectif #
            
δ_res = mean(mean(metrics.renewable_share, dims = 2) .< mg.parameters.renewable_share)
           
λ1 = 1e6

sum(metrics.cost.total[:,1])
return metrics.npv.total[1,1] - λ1 * δ_res