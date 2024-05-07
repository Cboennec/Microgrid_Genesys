#=
    Designer based on a metaheuristic
=#


mutable struct MetaheuristicOptions
    method::AbstractMetaheuristic
    iterations::Int64
    multithreads::Bool
    controller::AbstractController
    isnpv::Bool
    reducer::AbstractScenariosReducer
    objective_risk::AbstractRiskMeasure
    share_risk::AbstractRiskMeasure
    lpsp_risk::AbstractRiskMeasure
    lpsp_tol::Float64
    reopt::Bool
    read_reduction::Union{String, Nothing}
    write_reduction::Union{String, Nothing}

    MetaheuristicOptions(; method = Clearing(),
                           iterations = 50,
                           multithreads = false,
                           controller = RBC(),
                           isnpv = false,
                           reducer = FeatureBasedReducer(),
                           objective_risk = Expectation(),
                           share_risk = Expectation(),
                           lpsp_risk = WorstCase(),
                           lpsp_tol = 1e-3,
                           reopt = false,
                           read_reduction = nothing,
                           write_reduction = nothing) =
                           new(method, iterations, multithreads, controller, isnpv, reducer, objective_risk, share_risk, lpsp_risk, lpsp_tol, reopt, read_reduction, write_reduction)

end

mutable struct Metaheuristic <: AbstractDesigner
    options::MetaheuristicOptions

    generations::Dict
    storages::Dict
    converters::Dict
    subscribed_power::Dict
    
    decisions::NamedTuple
    results::MetaheuristicResults
    history::AbstractScenarios

 

    Metaheuristic(; options = MetaheuristicOptions(),  generations = Dict(), storages = Dict(), converters = Dict(), subscribed_power = Dict()) = new(options, generations, storages, converters, subscribed_power)
end


# Objective functions
function fobj(decisions::Array{Float64,1}, mg::Microgrid, designer::Metaheuristic, ω::Scenarios, probabilities::Array{Float64})
    # Paramters
    nh, ny, ns = size(ω.demands[1].power)
    λ1 = λ2 = λ3 = 1e6

    # Initialize mg
    mg_m = deepcopy(mg)

    # Initialize controller
    controller_m = initialize_controller!(mg_m, designer.options.controller, ω)

    # Initialize with the manual designer
    designer_m = initialize_designer!(mg_m, Manual(generations = [decisions[1:length(mg_m.generations)]...], storages = [decisions[length(mg_m.generations)+1:length(mg_m.generations)+length(mg_m.storages)]...], converters = [decisions[end-length(mg_m.converters)+1:end]...]), ω)

    # Simulate
    simulate!(mg_m, controller_m, designer_m, ω)

    # Metrics
    metrics = Metrics(mg_m, designer_m)

    # Share constraint
    share = max(0., mg.parameters.renewable_share - conditional_value_at_risk([reshape(metrics.renewable_share[2:ny, 1:ns], :, 1)...],  probabilities,  designer.options.share_risk))

    # LPSP constraint for the heat
    metrics.lpsp.heat isa Nothing ? lpsp = 0. : lpsp = max(0., conditional_value_at_risk([reshape(metrics.lpsp.heat[2:ny, 1:ns], :, 1)...], probabilities,  designer.options.lpsp_risk) - designer.options.lpsp_tol)

    # SoC constraint for the seasonal storage
    soc_seasonal = 0.
    for a in mg_m.storages
        if a isa H2Tank
            soc_seasonal += sum(max(0., a.soc[1,y,s] - a.soc[end,y,s]) for y in 2:ny, s in 1:ns)
        end
    end

    # Objective - Algortihm find the maximum
    if designer.options.isnpv
        # NPV
        npv = conditional_value_at_risk([metrics.npv.total...], probabilities, designer.options.objective_risk)
        return npv - λ1 * share - λ2 * lpsp - λ3 * soc_seasonal
    else
        # Equivalent annual cost
        eac = conditional_value_at_risk([metrics.eac.total...], probabilities, designer.options.objective_risk)
        return - eac - λ1 * share - λ2 * lpsp - λ3 * soc_seasonal
    end
end

### Offline
function initialize_designer!(mg::Microgrid, designer::Metaheuristic, ω::Scenarios)
    # Preallocate and assigned values
    preallocate!(mg, designer)

    # Scenario reduction from the optimization scenario pool
    if designer.options.isnpv
        println("Starting scenario reduction...")
        ω_reduced, probabilities = reduce(designer.options.reducer, ω)
    else
        if isa(designer.options.read_reduction, Nothing)
            println("Starting scenario reduction...")
            ω_reduced, probabilities = reduce(designer.options.reducer, ω)
            # Saving
            if !isa(designer.options.write_reduction, Nothing)
                save(designer.options.write_reduction, "scenarios", ω_reduced, "probabilities", probabilities)
            end
        else
            println("Reading scenario reduction from file...")
            ω_reduced = load(designer.options.read_reduction, "scenarios")
            probabilities = load(designer.options.read_reduction, "probabilities")
        end
        # Repeat to simulate 2 years
        ω_reduced = repeat(ω_reduced, 1, 2, 1)
    end

    # Bounds
    lb, ub = set_bounds(mg)

    # Optimize
    designer.results = optimizeMetaheuristic(lb, ub,
                                               designer.options.method,
                                               options = MetaResultOptions(iterations = designer.options.iterations, multithreads = designer.options.multithreads)
    ) do decisions
        fobj(decisions, mg, designer, ω_reduced, probabilities)
      end

    # Assign values
    for k in 1:length(mg.generations)
        designer.decisions.generations[k][1,:] .= designer.results.minimizer[k]
    end
    for k in 1:length(mg.storages)
        designer.decisions.storages[k][1,:] .= designer.results.minimizer[length(mg.generations)+k]
    end
    for k in 1:length(mg.converters)
        designer.decisions.converters[k][1,:] .= designer.results.minimizer[end-length(mg.converters)+k]
    end

    # Save history for online optimization
    designer.history = ω_reduced

    return designer
end




# Objective functions
function fobj2(decisions::Array{Float64,1}, mg::Microgrid, designer::Metaheuristic, ω::Scenarios, varID::Dict)
    # Paramters
    nh, ny, ns = size(ω.demands[1].power)
    λ_eq = 1e6
    λ_res = 1e8


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

    subscribed_power = Dict("Electricity" => 10.)

    designer_m = initialize_designer!(mg_m, Manual(generations = generations, storages = storages, converters = converters, subscribed_power = subscribed_power), ω)

    # Initialize controller
    controller_m = initialize_controller!(mg_m, designer.options.controller, ω)

    # Simulate
    simulate!(mg_m, controller_m, designer_m, ω, options = Options(mode = "serial"))

    # Metrics
    metrics = Metrics(mg_m, designer_m)


    

    δ_res = mean(min.(metrics.renewable_share .- mg_m.parameters.renewable_share, 0.))

    δ_eq = get_δ_eq(mg_m)/(ny*ns)   

    
    return metrics.npv.total[1,1] - λ_res * δ_res - λ_eq * δ_eq 
    
end




# Objective functions
function fobj3(decisions::Array{Float64,1}, mg::Microgrid, designer::Metaheuristic, ω::Scenarios, varID::Dict; RES_target, design_decisions_ref, lb_design, ub_design)
    # Paramters
    nh, ny, ns = size(ω.demands[1].power)
    λeq = 1e6
    λres = 1e7 #1% => 10000
    λvar1 = 5e6 #norm dist 1% = 5000
    λvar2 = 5e6 #1% decision range = 5000

    ϵ_res = 0.01 #1% res threshold
    ϵ_var = 0.01 #1% var threshold

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
        if a isa Solar
            generations[string(typeof(a))] = decisions[varID[string(typeof(a))]]
        end
    end
    for a in mg_m.storages
        storages[string(typeof(a))] = decisions[varID[string(typeof(a))]]  
    end

    subscribed_power = Dict("Electricity" => 10.)

    #Model configuration 
    liion = get_liion_model_config((eff = decisions[varID["Battery model"].eff], soh = decisions[varID["Battery model"].soh], couplage = decisions[varID["Battery model"].couplage]))
    elyz = get_electrolyer_model_config((eff = decisions[varID["Electrolyzer model"].eff], soh = decisions[varID["Electrolyzer model"].soh], couplage = decisions[varID["Electrolyzer model"].couplage]))
    fc = get_fuelcell_model_config((eff = decisions[varID["FuelCell model"].eff], soh = decisions[varID["FuelCell model"].soh], couplage = decisions[varID["FuelCell model"].couplage]))
    mg_m.storages[1] = preallocate!(liion, nh, ny, ns)
    mg_m.converters[1] = preallocate!(elyz, nh, ny, ns)
    mg_m.converters[2] = preallocate!(fc, nh, ny, ns)

    designer_m = initialize_designer!(mg_m, Manual(generations = generations, storages = storages, converters = converters, subscribed_power = subscribed_power), ω_a)

    # Initialize controller
    controller_m = initialize_controller!(mg_m, designer.options.controller, ω)

    # Simulate
    simulate!(mg_m, controller_m, designer_m, ω, options = Options(mode = "serial"))

    # Metrics
    metrics = Metrics(mg_m, designer_m)

    δres = max(0, mean(abs(mean(metrics.renewable_share) - RES_target) - ϵ_res))
       
    δeq = get_δ_eq(mg_m)   
    
    #All design decision (non related to model decisions)
    design_tuples = [val for (key, val) in pairs(varID) if !occursin("model", key)]
    design_ids = []
    for id_tuples in design_tuples
        for id in id_tuples
            push!(design_ids, id)
        end
    end

    range_design_decisions = ub_design .- lb_design

    δvar2 = get_δ_var2(decisions[sort(design_ids)], design_decisions_ref, range_design_decisions, ϵ_var) 
    
    δX_ref = get_δ_X_ref(decisions) #Same model as reference
    
    return metrics.npv.total[1,1] - λres * δres - λeq * δeq - λvar2 * δvar2 - δX_ref * 1e12
    
end


### Online
# Loi de gestion d'investissement dynamique
# ex : remplacer la batterie par une equivalente à partir d'un seuil défini
# regarder dans le papier sur investissement dynamique
function compute_investment_decisions!(y::Int64, s::Int64, mg::Microgrid, designer::Metaheuristic)

    for a in mg.storages
        if a isa Liion
            if a.soh[end,y,s] <= a.SoH_threshold
                designer.decisions.storages["Liion"][y,s] =  designer.storages["Liion"]
            end
        elseif a isa H2Tank
            #No SoH yet for H2Tank
        end
    end

    for a in mg.converters
        if a isa FuelCell
            if a.soh[end,y,s] <= a.SoH_threshold
                designer.decisions.converters["FuelCell"].surface[y,s] = designer.converters["FuelCell"].surface
                designer.decisions.converters["FuelCell"].N_cell[y,s] = designer.converters["FuelCell"].N_cell     
            else
                designer.decisions.converters["FuelCell"].surface[y,s] = 0.
                designer.decisions.converters["FuelCell"].N_cell[y,s] = 0.
            end
        elseif a isa Electrolyzer
            if a.soh[end,y,s] <= a.SoH_threshold
                designer.decisions.converters["Electrolyzer"].surface[y,s] = designer.converters["Electrolyzer"].surface
                designer.decisions.converters["Electrolyzer"].N_cell[y,s] = designer.converters["Electrolyzer"].N_cell
            else
                designer.decisions.converters["Electrolyzer"].surface[y,s] = 0.
                designer.decisions.converters["Electrolyzer"].N_cell[y,s] = 0.
            end
        end
    end

    return nothing

end


function get_decision_keys_name(mg::Microgrid)
    assets = Dict()
    # Assign values
    for a in mg.generations
        assets[string(typeof(a))] = ""
    end
    for a in mg.storages
        assets[string(typeof(a))] = ""
    end
    for a in mg.converters
        assets[string(typeof(a))] = ""
    end
    for a in mg.grids
        assets[string(typeof(a.carrier))] = ""
    end

    return keys(assets)
end

### Offline
function initialize_designer!(mg::Microgrid, designer::Metaheuristic, ω::Scenarios, ub::Vector{Float64}, lb::Vector{Float64}, varID::Dict; f_obj = fobj2)
    
    @assert(get_decision_keys_name(mg) == keys(varID), string("VarID keys doesnt correspond to asset list \n", get_decision_keys_name(mg), " != ",  keys(varID) ))
    # Preallocate and assigned values
    preallocate!(mg, designer)

    #Scenarios reduction #Currently removed   

    # Optimize
    designer.results = optimizeMetaheuristic(lb, ub,
                                               designer.options.method,
                                               options = MetaResultOptions(iterations = designer.options.iterations, multithreads = designer.options.multithreads)
    ) do decisions
        f_obj(decisions, mg, designer, ω, varID)
      end



    # Assign values
    for a in mg.generations
        if a isa Solar
            designer.generations[string(typeof(a))] = designer.results.minimizer[varID[string(typeof(a))]]
        end
    end
    for a in mg.storages
            designer.storages[string(typeof(a))] = designer.results.minimizer[varID[string(typeof(a))]]
    end
    for a in mg.converters
        if a isa Electrolyzer
            designer.converters[string(typeof(a))] = (N_cell = Int(round(designer.results.minimizer[varID[string(typeof(a))].N_cell])), surface = designer.results.minimizer[varID[string(typeof(a))].surface])
        elseif a isa FuelCell
            designer.converters[string(typeof(a))] = (N_cell = Int(round(designer.results.minimizer[varID[string(typeof(a))].N_cell])), surface = designer.results.minimizer[varID[string(typeof(a))].surface])
        elseif a isa FuelCell
            designer.converters[string(typeof(a))] = designer.results.minimizer[varID[string(typeof(a))]]
        end
    end
    for a in mg.grids
        designer.subscribed_power[string(typeof(a.carrier))] = designer.results.minimizer[varID[string(typeof(a.carrier))]] 
        designer.decisions.subscribed_power[string(typeof(a.carrier))][:,:] .= designer.results.minimizer[varID[string(typeof(a.carrier))]] 
    end
    

    # Save history for online optimization
    designer.history = ω

    return designer
end



# Designer for multi objective (a vector of designer is returned) 
# A selection is then needed (by ID or by using GraphicalDesignSelection )
function generate_designers_MO(mg::Microgrid, designer::Metaheuristic, ω::Scenarios, ub::Vector{Float64}, lb::Vector{Float64}, varID::Dict; f_obj = fobj2)
    
    @assert(get_decision_keys_name(mg) == keys(varID), string("VarID keys doesnt correspond to asset list \n", get_decision_keys_name(mg), " != ",  keys(varID) ))
    # Preallocate and assigned values
    preallocate!(mg, designer)

    #Scenarios reduction #Currently removed   

    # Optimize
    designer.results = optimizeMetaheuristic(lb, ub,
                                               designer.options.method,
                                               options = MetaResultOptions(iterations = designer.options.iterations, multithreads = designer.options.multithreads)
    ) do decisions
        f_obj(decisions, mg, designer, ω, varID)
      end

    return designer, designer.results.sensitivity
end

function initialize_designer_MO(mg::Microgrid, results::Metaheuristic, varID::Dict, index::Int)

    designer = Metaheuristic(options = MetaheuristicOptions(;method = results.results.method))
    preallocate!(mg, designer)


    solution = results.results.population[index]
    solution.val_param
    # Assign values
    for a in mg.generations
        if a isa Solar
            designer.generations[string(typeof(a))] = solution.val_param[varID[string(typeof(a))]]
        end
    end
    for a in mg.storages
            designer.storages[string(typeof(a))] = solution.val_param[varID[string(typeof(a))]]
    end
    for a in mg.converters
        if a isa Electrolyzer
            designer.converters[string(typeof(a))] = (N_cell = Int(round(solution.val_param[varID[string(typeof(a))].N_cell])), surface = solution.val_param[varID[string(typeof(a))].surface])
        elseif a isa FuelCell
            designer.converters[string(typeof(a))] = (N_cell = Int(round(solution.val_param[varID[string(typeof(a))].N_cell])), surface = solution.val_param[varID[string(typeof(a))].surface])
        elseif a isa FuelCell
            designer.converters[string(typeof(a))] = solution.val_param[varID[string(typeof(a))]]
        end
    end
    for a in mg.grids
        designer.subscribed_power[string(typeof(a.carrier))] = solution.val_param[varID[string(typeof(a.carrier))]] 
        designer.decisions.subscribed_power[string(typeof(a.carrier))][:,:] .= solution.val_param[varID[string(typeof(a.carrier))]] 
    end
    

    # Save history for online optimization
    designer.history = ω

    designer.results = results.results
    
    return designer


end

### Online
# Loi de gestion d'investissement dynamique
# ex : remplacer la batterie par une equivalente à partir d'un seuil défini
# regarder dans le papier sur investissement dynamique
function compute_investment_decisions!(y::Int64, s::Int64, mg::Microgrid, designer::Metaheuristic)

    for a in mg.storages
        if a isa Liion
            if a.soh[end,y,s] <= a.SoH_threshold
                designer.decisions.storages[string(typeof(a))][y,s] = designer.storages[string(typeof(a))]
            end
        elseif a isa H2Tank
            #No SoH yet for H2Tank
        end
    end

    for a in mg.converters
        if a isa FuelCell
            if a.soh[end,y,s] <= a.SoH_threshold
                designer.decisions.converters[string(typeof(a))].surface[y,s] = designer.converters[string(typeof(a))].surface
                designer.decisions.converters[string(typeof(a))].N_cell[y,s] = designer.converters[string(typeof(a))].N_cell     
            else
                designer.decisions.converters[string(typeof(a))].surface[y,s] = 0.
                designer.decisions.converters[string(typeof(a))].N_cell[y,s] = 0.
            end
        elseif a isa Electrolyzer
            if a.soh[end,y,s] <= a.SoH_threshold
                designer.decisions.converters[string(typeof(a))].surface[y,s] = designer.converters[string(typeof(a))].surface
                designer.decisions.converters[string(typeof(a))].N_cell[y,s] = designer.converters[string(typeof(a))].N_cell
            else
                designer.decisions.converters[string(typeof(a))].surface[y,s] = 0.
                designer.decisions.converters[string(typeof(a))].N_cell[y,s] = 0.
            end
        end
    end

    return nothing

end

### Utils
function set_bounds(mg::Microgrid)
    # Initialization
    lb, ub = [], []
    # Generations
    for a in mg.generations
        push!(lb, a.bounds.lb)
        push!(ub, a.bounds.ub)
    end
    # Storages
    for a in mg.storages
        push!(lb, a.bounds.lb)
        push!(ub, a.bounds.ub)
    end
    # Converters
    for a in mg.converters
        push!(lb, a.bounds.lb)
        push!(ub, a.bounds.ub)
    end
    return lb, ub
end



function get_δ_eq(mg::Microgrid, type::Type)

    δ_eq = 0

    nh, ny, ns = mg.parameters.nh, mg.parameters.ny, mg.parameters.ns


    energy_carriers_list = []
    for conv in mg.converters
        for carrier in conv.carrier
            push!(energy_carriers_list, carrier)
        end
    end
    for demand in mg.demands
        push!(energy_carriers_list,  demand.carrier)
    end


    tot = zeros(nh , ny, ns)


    for a in mg.demands
        if a.carrier isa type
            tot .+= -a.carrier.power
        end
    end
    # Generations
    for a in mg.generations
        if a.carrier isa type
            tot .+= a.carrier.power
        end
    end
    # Storages
    for a in mg.storages
        if a.carrier isa type
            tot .+= a.carrier.power
        end
    end
    # Converters
    for a in mg.converters
        for c in a.carrier
            if c isa type
                tot .+= c.power
            end
        end
    end
    for a in mg.grids
        if a.carrier isa type
            tot  .+= a.carrier.power
        end
    end
    if type == Electricity
        δ_eq += sum(tot .!= 0)
    elseif type == Heat 
        δ_eq += sum(tot .< 0)
    elseif type == Hydrogen
        δ_eq += sum(tot .!= 0)
    end
    
    return δ_eq
end


function get_liion_model_config(decisions::NamedTuple{(:eff, :soh, :couplage), Tuple{Float64, Float64, Float64}})

    if Int(round(decisions.eff)) == 0 
        eff_model = LinearLiionEfficiency()
    elseif  Int(round(decisions.eff)) == 1
        eff_model = PolynomialLiionEfficiency()
    else
        println("problem with var liion eff model : ", decisions.eff)
    end

    if Int(round(decisions.soh)) == 0 
        soh_model =  FixedLifetimeLiion()
    elseif  Int(round(decisions.soh)) == 1
        soh_model =  EnergyThroughputLiion(;nCycle_ini = Int(floor(fatigue_data.cycle[findfirst(fatigue_data.DoD .> (0.6))])))
    elseif  Int(round(decisions.soh)) == 2 
        soh_model =  RainflowLiion(fatigue_data = fatigue_data)
    elseif  Int(round(decisions.soh)) == 3 
        soh_model =  SemiEmpiricalLiion()
    else
        println("problem with var liion soh model : ", decisions.soh)
    end

    if Int(round(decisions.couplage)) == 0
        coupl = (E=false, R=false)
    elseif  Int(round(decisions.couplage)) == 1
        coupl = (E=true, R=true)
    else
        println("problem with var liion couple model : ", decisions.couplage)
    end


    
    return Liion(SoC_model = eff_model, SoH_model = soh_model, couplage = coupl)

end


function get_electrolyer_model_config(decisions::NamedTuple{(:eff, :soh, :couplage), Tuple{Float64, Float64, Float64}})
   
    if Int(round(decisions.eff)) == 0 
        eff_model = FixedElectrolyzerEfficiency()
    elseif Int(round(decisions.eff)) == 1
        eff_model = LinearElectrolyzerEfficiency()
    elseif Int(round(decisions.eff)) == 2
        eff_model = PolarizationElectrolyzerEfficiency()
    else
        println("problem with var Electrolyzer eff model : ", decisions.eff)
    end


    if Int(round(decisions.soh)) == 0 
        soh_model =  FixedLifetimeElectrolyzer()
    elseif  Int(round(decisions.soh)) == 1
        soh_model =  FunctHoursAgingElectrolyzer()
    else
        println("problem with var Electrolyzer soh model : ", decisions.soh)
    end


    if Int(round(decisions.couplage)) == 0
        coupl = false
    elseif  Int(round(decisions.couplage)) == 1
        coupl = true
    else
        println("problem with var Electrolyzer couple model : ", decisions.couplage)
    end


    
    return Electrolyzer(;V_J_ini = V_J_Elyz, EffModel = eff_model, SoH_model = soh_model, couplage = coupl)

end



function get_fuelcell_model_config(decisions::NamedTuple{(:eff, :soh, :couplage), Tuple{Float64, Float64, Float64}})
   
    if Int(round(decisions.eff)) == 0 
        eff_model = FixedFuelCellEfficiency()
    elseif Int(round(decisions.eff)) == 1
        eff_model = LinearFuelCellEfficiency()
    elseif Int(round(decisions.eff)) == 2
        eff_model = PolarizationFuelCellEfficiency()
    else
        println("problem with var Fuel Cell eff model : ", decisions.eff)
    end


    if Int(round(decisions.soh)) == 0 
        soh_model =  FixedLifetimeFuelCell()
    elseif Int(round(decisions.soh)) == 1
        soh_model =  FunctHoursAgingFuelCell(;deg_params=deg, J_base = 0.1)
    elseif Int(round(decisions.soh)) == 2
        soh_model =  PowerAgingFuelCell(;deg_params=deg, StartStop = false)
    elseif Int(round(decisions.soh)) == 3
        soh_model =  PowerAgingFuelCell(;deg_params=deg, StartStop = true)
    else
        println("problem with var Fuel Cell soh model : ", decisions.soh)
    end

    if Int(round(decisions.couplage)) == 0
        coupl = false
    elseif Int(round(decisions.couplage)) == 1
        coupl = true
    else
        println("problem with var Fuel Cell couple model : ", decisions.couplage)
    end

    
    return FuelCell(;V_J_ini = V_J_FC, EffModel = eff_model, SoH_model = soh_model, couplage = coupl)

end


function get_δ_X_ref(decisions)
    #Test if all model variables are activating the reference model. this model should not be considered.
    if mean((Int.(round.((decisions[1:9]))) .+ 0.5) .>= ub[1:9]) == 1
        return true
    else
        return false
    end
end

#Take the maximum normalize difference in the decisions between the ref and the tested solution.
# Then apply a threshold of ϵ_var below which it's brought to 0.
function get_δ_var2(design_decisions, design_decisions_ref, range_design_decisions, ϵ_var) 
    return max(maximum(abs.(design_decisions .- design_decisions_ref) ./ range_design_decisions) - ϵ_var, 0)
end