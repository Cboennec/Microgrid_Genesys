#=
    Manual designer
=#

mutable struct Manual <: AbstractDesigner
    generations::Dict
    storages::Dict
    converters::Dict
    subscribed_power::Dict
    decisions::NamedTuple

    Manual(; generations = Dict(), storages = Dict(), converters = Dict(), subscribed_power = Dict()) = new(generations, storages, converters, subscribed_power)
end

### Offline
function initialize_designer!(mg::Microgrid, designer::Manual, ω::AbstractScenarios)
    # Preallocation
    preallocate!(mg, designer)

    # Fix initial values

    # for k in 1:length(mg.generations)
    #     if mg.generations[k] isa Solar
    #         designer.decisions.generations["PV"][1,:] .= designer.generations["PV"]
    #     end
    # end
    # for k in 1:length(mg.storages)
    #     if mg.storages[k] isa Liion
    #         designer.decisions.storages["Liion"][1,:] .= designer.storages["Liion"]
    #     elseif mg.storages[k] isa H2Tank
    #         designer.decisions.storages["H2Tank"][1,:] .= designer.storages["H2Tank"]
    #     end
    # end

    # for k in 1:length(mg.converters)
    #     if mg.converters[k] isa FuelCell
    #         key = "FuelCell"
    #         designer.decisions.converters[key].surface[1,:] .= designer.converters[key].surface
    #         designer.decisions.converters[key].N_cell[1,:] .= designer.converters[key].N_cell
    #     elseif mg.converters[k] isa Electrolyzer
    #         key = "Electrolyzer"
    #         designer.decisions.converters[key].surface[1,:] .= designer.converters[key].surface
    #         designer.decisions.converters[key].N_cell[1,:] .= designer.converters[key].N_cell
    #     elseif mg.converters[k] isa Heater
    #         key = "Heater"
    #         designer.decisions.converters[key][1,:] .= designer.converters[key]
    #     end
    # end

    #The value is the same for all years
    for k in 1:length(mg.grids)
        if mg.grids[k].carrier isa Electricity
            designer.decisions.subscribed_power["Electricity"][:,:] .= designer.subscribed_power["Electricity"]
        elseif mg.grids[k].carrier isa Heat
            designer.decisions.subscribed_power["Heat"][:,:] .= designer.subscribed_power["Heat"]
        elseif mg.grids[k].carrier isa Hydrogen
            designer.decisions.subscribed_power["Hydrogen"][:,:] .= designer.subscribed_power["Hydrogen"]
        end
    end

    return designer
end

### Online
function compute_investment_decisions!(y::Int64, s::Int64, mg::Microgrid, designer::Manual)

    
    for (k,a) in enumerate(mg.storages)
        if hasproperty(a, :soh)
            if a.soh[end,y,s] <= a.SoH_threshold
                designer.decisions.storages[string(typeof(a))][y,s] = designer.storages[string(typeof(a))]
            else 
                designer.decisions.storages[string(typeof(a))][y,s] = 0.
            end
        else
            designer.decisions.storages[string(typeof(a))][y,s] = 0.
        end
    end
   
    #Compute decision
    for k in 1:length(mg.converters)
        if mg.converters[k] isa FuelCell
            if mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold
                #println( "mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold : " ,  mg.converters[k].soh[end,y,s], " <= " ,mg.converters[k].SoH_threshold)
                designer.decisions.converters["FuelCell"].surface[y,s] = designer.converters["FuelCell"].surface
                designer.decisions.converters["FuelCell"].N_cell[y,s] = designer.converters["FuelCell"].N_cell
            else
                designer.decisions.converters["FuelCell"].surface[y,s] = 0.
                designer.decisions.converters["FuelCell"].N_cell[y,s] = 0.
            end
        elseif mg.converters[k] isa Electrolyzer

            if mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold
                #println( "mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold : " ,  mg.converters[k].soh[end,y,s], " <= " ,mg.converters[k].SoH_threshold)
                designer.decisions.converters["Electrolyzer"].surface[y,s] = designer.converters["Electrolyzer"].surface
                designer.decisions.converters["Electrolyzer"].N_cell[y,s] = designer.converters["Electrolyzer"].N_cell
            else
                designer.decisions.converters["Electrolyzer"].surface[y,s] = 0.
                designer.decisions.converters["Electrolyzer"].N_cell[y,s] = 0.
            end
        elseif mg.converters[k] isa Heater
            key = "Heater"
        end
    end

    return nothing
end
