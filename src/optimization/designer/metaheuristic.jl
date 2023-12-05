#=
    Designer based on a metaheuristic
=#

mutable struct MetaheuristicOptions
    method::Metaheuristics.AbstractMetaheuristic
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

    MetaheuristicOptions(; method = Metaheuristics.Clearing(),
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
    results::Metaheuristics.MetaheuristicResults
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
    designer.results = Metaheuristics.optimize(lb, ub,
                                               designer.options.method,
                                               options = Metaheuristics.Options(iterations = designer.options.iterations, multithreads = designer.options.multithreads)
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
    λ1 = λ2 = 1e6


    # Initialize mg
    mg_m = deepcopy(mg)
    mg_m.parameters = GlobalParameters(nh, ny, ns)


    # Initialize with the manual designer
    converters, generations, storages, subscribed_power = Dict(), Dict(), Dict(), Dict()
    for a in mg_m.converters
        if a isa Electrolyzer
            converters["Electrolyzer"] = (surface = decisions[varID["Electrolyzer"].surface], N_cell=Int64(round(decisions[varID["Electrolyzer"].N_cell])))
        elseif a isa FuelCell 
            converters["FuelCell"] = (surface = decisions[varID["FuelCell"].surface], N_cell=Int64(round(decisions[varID["FuelCell"].N_cell])))
        elseif a isa Heater
            converters["Heater"] = decisions[varID["Heater"]]
        end
    end
    for a in mg_m.generations
        if a isa Solar
            generations["PV"] = decisions[varID["PV"]]
        end
    end
    for a in mg_m.storages
        if a isa H2Tank
            storages["H2Tank"] = decisions[varID["H2Tank"]]  
        elseif a isa Liion
            storages["Liion"] = decisions[varID["Liion"]]  
        end
    end

    subscribed_power = Dict("Electricity" => 10.)

    designer_m = initialize_designer!(mg_m, Manual(generations = generations, storages = storages, converters = converters, subscribed_power = subscribed_power), ω_a)

    # Initialize controller
    controller_m = initialize_controller!(mg_m, designer.options.controller, ω)

    # Simulate
    simulate!(mg_m, controller_m, designer_m, ω, options = Options(mode = "serial"))

    # Metrics
    metrics = Metrics(mg_m, designer_m)

    δ_res = mean(mean(metrics.renewable_share, dims = 2) .< mg.parameters.renewable_share)
       
    δ_eq = get_δ_eq(mg_m)    
    
    return metrics.npv.total[1,1] - λ1 * δ_res - λ2 * δ_eq 
    
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



### Offline
function initialize_designer!(mg::Microgrid, designer::Metaheuristic, ω::Scenarios, ub::Vector{Float64}, lb::Vector{Float64}, varID::Dict)
    # Preallocate and assigned values
    preallocate!(mg, designer)

    #Scenarios reduction #Currently removed   

    # Optimize
    designer.results = Metaheuristics.optimize(lb, ub,
                                               designer.options.method,
                                               options = Metaheuristics.Options(iterations = designer.options.iterations, multithreads = designer.options.multithreads)
    ) do decisions
        fobj2(decisions, mg, designer, ω, varID)
      end



    # Assign values
    for a in mg.generations
        if a isa Solar
            designer.generations["PV"] = designer.results.minimizer[varID["PV"]]
        end
    end
    for a in mg.storages
        if a isa Liion
            designer.storages["Liion"] = designer.results.minimizer[varID["Liion"]]
        elseif a isa H2Tank
            designer.storages["H2Tank"] = designer.results.minimizer[varID["H2Tank"]]
        end
    end
    for a in mg.converters
        if a isa Electrolyzer
            designer.converters["Electrolyzer"] = (N_cell = designer.results.minimizer[varID["Electrolyzer"].N_cell], surface = designer.results.minimizer[varID["Electrolyzer"].surface])
        elseif a isa FuelCell
            designer.converters["FuelCell"] = (N_cell = designer.results.minimizer[varID["FuelCell"].N_cell], surface = designer.results.minimizer[varID["FuelCell"].surface])
        elseif a isa FuelCell
            designer.converters["Heater"] = designer.results.minimizer[varID["Heater"]]
        end
    end

    # Save history for online optimization
    designer.history = ω

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



function get_δ_eq(mg::Microgrid)

    δ_eq = 0


    energy_carriers_list = []
    for conv in mg.converters
        for carrier in conv.carrier
            push!(energy_carriers_list, carrier)
        end
    end
    for demand in mg.demands
        push!(energy_carriers_list,  demand.carrier)
    end

    energy_carriers = unique((typeof(a) for a in energy_carriers_list))

    tot = zeros(length(energy_carriers), nh , ny, ns)
    for (i, type) in enumerate(energy_carriers)
        # Demands
        for (k, a) in enumerate(mg.demands)
            if a.carrier isa type
                tot[i,:,:,:] .+= -a.carrier.power[:,:,:]
            end
        end
        # Generations
        for (k, a) in enumerate(mg.generations)
            if a.carrier isa type
                tot[i,:,:,:] .+= a.carrier.power[:,:,:]
            end
        end
        # Storages
        for (k, a) in enumerate(mg.storages)
            if a.carrier isa type
                tot[i,:,:,:] .+= a.carrier.power[:,:,:]
            end
        end
        # Converters
        for (k, a) in enumerate(mg.converters)
            for c in a.carrier
                if c isa type
                    tot[i,:,:,:] .+= c.power[:,:,:]
                end
            end
        end
        for (k, a) in enumerate(mg.grids)
            if a.carrier isa type
                tot[i,:,:,:] .+= a.carrier.power[:,:,:]
            end
        end
        if type == Electricity
            δ_eq += sum(tot[i,:,:,:] .!= 0)
        elseif type == Heat 
            δ_eq += sum(tot[i,:,:,:] .< 0)
        elseif type == Hydrogen
            δ_eq += sum(tot[i,:,:,:] .!= 0)
        end
    end
    return δ_eq
end