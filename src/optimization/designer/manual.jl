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

    for k in 1:length(mg.generations)
        if mg.generations[k] isa Solar
            designer.decisions.generations["PV"][1,:] .= designer.generations["PV"]
        end
    end
    for k in 1:length(mg.storages)
        if typeof(mg.storages[k]) <: AbstractLiion
            designer.decisions.storages["Liion"][1,:] .= designer.storages["Liion"]
        elseif mg.storages[k] isa H2Tank
            designer.decisions.storages["H2Tank"][1,:] .= designer.storages["H2Tank"]
        end
    end

    for k in 1:length(mg.converters)
        if typeof(mg.converters[k]) <: AbstractFuelCell
            key = "FuelCell"
            if mg.converters[k] isa FuelCell_V_J || mg.converters[k] isa FuelCell_lin
                designer.decisions.converters[key].surface[1,:] .= designer.converters[key].surface
                designer.decisions.converters[key].N_cell[1,:] .= designer.converters[key].N_cell
            else 
                designer.decisions.converters[key][1,:] .= designer.converters[key]
            end
        elseif mg.converters[k] isa Electrolyzer
            key = "Electrolyzer"
            designer.decisions.converters[key][1,:] .= designer.converters[key]
        elseif mg.converters[k] isa Heater
            key = "Heater"
            designer.decisions.converters[key][1,:] .= designer.converters[key]
        end
    end

    for k in 1:length(mg.grids)
        if mg.grids[k].carrier isa Electricity
            designer.decisions.subscribed_power["Electricity"][1,:] .= designer.subscribed_power["Electricity"]
        elseif mg.grids[k].carrier isa Heat
            designer.decisions.subscribed_power["Heat"][1,:] .= designer.subscribed_power["Heat"]
        elseif mg.grids[k].carrier isa Hydrogen
            designer.decisions.subscribed_power["Hydrogen"][1,:] .= designer.subscribed_power["Hydrogen"]
        end
    end

    return designer
end

### Online
function compute_investment_decisions!(y::Int64, s::Int64, mg::Microgrid, designer::Manual)

    for k in 1:length(mg.storages)
        if typeof(mg.storages[k]) <: AbstractLiion
            if mg.storages[k].soh[end,y,s] <= mg.storages[k].SoH_threshold
                designer.decisions.storages["Liion"][y,s] = designer.storages["Liion"]
            else 
                designer.decisions.storages["Liion"][y,s] = 0.
            end
        end
    end
   
    #Compute decision
    for k in 1:length(mg.converters)
        if typeof(mg.converters[k]) <: AbstractFuelCell
            if mg.converters[k] isa FuelCell_V_J || mg.converters[k] isa FuelCell_lin

                if mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold
                    #println( "mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold : " ,  mg.converters[k].soh[end,y,s], " <= " ,mg.converters[k].SoH_threshold)
                    designer.decisions.converters["FuelCell"].surface[y,s] = designer.converters["FuelCell"].surface
                    designer.decisions.converters["FuelCell"].N_cell[y,s] = designer.converters["FuelCell"].N_cell
                else
                    designer.decisions.converters["FuelCell"].surface[y,s] = 0.
                    designer.decisions.converters["FuelCell"].N_cell[y,s] = 0.
                end
            else 
                key = "FuelCell"
            end
        elseif typeof(mg.converters[k]) <: AbstractElectrolyzer
            if mg.converters[k] isa Electrolyzer_V_J

                if mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold
                    #println( "mg.converters[k].soh[end,y,s] <= mg.converters[k].SoH_threshold : " ,  mg.converters[k].soh[end,y,s], " <= " ,mg.converters[k].SoH_threshold)
                    designer.decisions.converters["Electrolyzer"].surface[y,s] = designer.converters["Electrolyzer"].surface
                    designer.decisions.converters["Electrolyzer"].N_cell[y,s] = designer.converters["Electrolyzer"].N_cell
                else
                    designer.decisions.converters["Electrolyzer"].surface[y,s] = 0.
                    designer.decisions.converters["Electrolyzer"].N_cell[y,s] = 0.
                end
            else 
                key = "Electrolyzer"
            end
        elseif mg.converters[k] isa Heater
            key = "Heater"
        end
    end

    return nothing
end
