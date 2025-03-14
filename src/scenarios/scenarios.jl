#=
    Scenario reduction functions
=#
abstract type AbstractScenarios end

"""
    mutable struct Scenarios{T, O, I} <: AbstractScenarios

A mutable struct representing Scenarios, which is a subtype of `AbstractScenarios`.

# Fields
- `demands::Vector{NamedTuple{(:t, :power), Tuple{T, O}}}`: A vector of named tuples representing the time and power demand.
- `generations::Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}`: A vector of named tuples representing the time, power generation, and cost.
- `storages::Vector{NamedTuple{(:cost,), Tuple{I}}}`: A vector of named tuples representing the cost of storage.
- `converters::Vector{NamedTuple{(:cost,), Tuple{I}}}`: A vector of named tuples representing the cost of converters.
- `grids::Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}`: A vector of named tuples representing the input cost, output cost, and exceeding cost of the grid.


"""
# mutable struct Scenarios{T, O, I} <: AbstractScenarios
#     demands::Vector{NamedTuple{(:t, :power),Tuple{T,O}}}
#     generations::Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}
#     storages::Vector{NamedTuple{(:cost,), Tuple{I}}}
#     converters::Vector{NamedTuple{(:cost,), Tuple{I}}}
#     grids::Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}
# end

mutable struct Scenarios <: AbstractScenarios
    demands::Vector{NamedTuple}
    generations::Vector{NamedTuple}
    storages::Vector{NamedTuple}
    converters::Vector{NamedTuple}
    grids::Vector{NamedTuple}
end


"""
    mutable struct MiniScenarios{T, O, I} <: AbstractScenarios

A mutable struct representing shortened Scenarios, which is a subtype of `AbstractScenarios`.
Only some days will be selected. This method implement the work of Van Der Heijde (DOI: 10.1016/j.apenergy.2019.04.030)

# Fields
- `demands::Vector{NamedTuple{(:t, :power), Tuple{T, O}}}`: A vector of named tuples representing the time and power demand.
- `generations::Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}`: A vector of named tuples representing the time, power generation, and cost.
- `storages::Vector{NamedTuple{(:cost,), Tuple{I}}}`: A vector of named tuples representing the cost of storage.
- `converters::Vector{NamedTuple{(:cost,), Tuple{I}}}`: A vector of named tuples representing the cost of converters.
- `grids::Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}`: A vector of named tuples representing the input cost, output cost, and exceeding cost of the grid.
- `days::Array{Int64,3}`: The matrix of the days used as representative indexed for each year and scenario  [day,year,sscenario]
- `sequence::Array{Int64,3}`: The matrix of the representative day used  to represent each day of the year indexed for each year and scenario  [day,year,scenario]
"""
mutable struct MiniScenarios <: AbstractScenarios
    demands::Vector{NamedTuple}
    generations::Vector{NamedTuple}
    storages::Vector{NamedTuple}
    converters::Vector{NamedTuple}
    grids::Vector{NamedTuple}
    days::Array{Int64, 3}
    sequence::Array{Int64, 3}
    scenarios_reconstructed::Scenarios
end
mutable struct MiniScenarios_my <: AbstractScenarios
    demands::Vector{NamedTuple}
    generations::Vector{NamedTuple}
    storages::Vector{NamedTuple}
    converters::Vector{NamedTuple}
    grids::Vector{NamedTuple}
    days::Array{Int64, 2}
    sequence::Array{Int64, 2}
    scenarios_reconstructed::Scenarios
end


"""
    function Scenarios(mg::Microgrid, d::Dict{})

Constructor function for creating a new `Scenarios` instance based on a given `Microgrid` and a `Dict` containing scenario data.

# Arguments
- `mg::Microgrid`: A Microgrid instance.
- `d::Dict{}`: A dictionary containing scenario data.

# Returns
- `Scenarios`: A Scenarios instance with the specified data.

## Example

```julia
microgrid = ...
scenario_data = ...

scenarios = Scenarios(microgrid, scenario_data)
```
"""
function Scenarios(mg::Microgrid, d::Dict{})
    # Utils to simplify the writting
    h, y, s = 1:mg.parameters.nh, 1:mg.parameters.ny, 1:mg.parameters.ns
    T, O, I = Array{DateTime,3}, Array{Float64, 3}, Array{Float64, 2}

    # Initialize
    demands = Vector{NamedTuple{(:t, :power),Tuple{T,O}}}(undef, length(mg.demands))
    generations = Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}(undef, length(mg.generations))
    storages = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.storages))
    converters = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.converters))
    grids = Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}(undef, length(mg.grids))
    # Demands
    for (k, a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            demands[k] = (t = d["ld_E"].t[h, y, s], power = d["ld_E"].power[h, y, s])
        elseif a.carrier isa Heat
            demands[k] = (t = d["ld_H"].t[h, y, s], power = d["ld_H"].power[h, y, s])
        end
    end
    # Generation
    for (k, a) in enumerate(mg.generations)
        if a isa Solar
            generations[k] = (t = d["pv"].t[h, y, s], power = d["pv"].power[h, y, s], cost = d["pv"].cost[y, s])
        end
    end
    # Storages
    for (k, a) in enumerate(mg.storages)

        if typeof(a) <: AbstractLiion
            storages[k] = (cost = d["liion"].cost[y, s],)
        elseif a isa ThermalStorage
            storages[k] = (cost = d["tes"].cost[y, s],)
        elseif a isa H2Tank
            storages[k] = (cost = d["h2tank"].cost[y, s],)
        end
    end
    # Converters
    for (k, a) in enumerate(mg.converters)
        if typeof(a) <: AbstractElectrolyzer
            converters[k] = (cost = d["elyz"].cost[y, s],)
        elseif typeof(a) <: AbstractFuelCell
            converters[k] = (cost = d["fc"].cost[y, s],)
        elseif a isa Heater
            converters[k] = (cost = d["heater"].cost[y, s],)
        end
    end
    # Grids
    for (k, a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            grids[k] = (cost_in = d["grid_Elec"].cost_in[h, y, s], cost_out = d["grid_Elec"].cost_out[h, y, s], cost_exceed = zeros(length(y),length(s)) .+ 10) #TODO this price should come from the scenarios
        end
    end

    return Scenarios(demands, generations, storages, converters, grids)
end





"""
    function Scenarios(mg::Microgrid, d::Dict{})

Constructor function for creating a new `Scenarios` instance based on a given `Microgrid` and a `Dict` containing scenario data.

# Arguments
- `mg::Microgrid`: A Microgrid instance.
- `d::Dict{}`: A dictionary containing scenario data.

# Returns
- `Scenarios`: A Scenarios instance with the specified data.

## Example

```julia
microgrid = ...
scenario_data = ...

scenarios = Scenarios(microgrid, scenario_data)
```
"""
function MiniScenarios(mg::Microgrid, ω::Scenarios, N_days::Int64; N_bins = 20, time_limit = [100,200], display_res = true)
    h, y, s = 1:mg.parameters.nh, 1:mg.parameters.ny, 1:mg.parameters.ns
    T, O, I = Array{DateTime,3}, Array{Float64, 3}, Array{Float64, 2}, Array{Int64, 3}


    demands = Vector{NamedTuple{(:t, :power),Tuple{T,O}}}(undef, length(mg.demands))
    generations = Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}(undef, length(mg.generations))
    storages = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.storages))
    converters = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.converters))
    grids = Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}(undef, length(mg.grids))

    demands_reconstructed = Vector{NamedTuple{(:t, :power),Tuple{T,O}}}(undef, length(mg.demands))
    generations_reconstructed = Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}(undef, length(mg.generations))
    storages_reconstructed = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.storages))
    converters_reconstructed = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.converters))
    grids_reconstructed = Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}(undef, length(mg.grids))

    days = Int.(zeros(N_days, mg.parameters.ny, mg.parameters.ns))
    sequence = Int.(zeros(365, mg.parameters.ny, mg.parameters.ns))

    index_hour = Int.(zeros(24*N_days, mg.parameters.ny, mg.parameters.ns))

    loads_E = zeros(length(h), length(y), length(s))
    loads_H = zeros(length(h), length(y), length(s))
    PVs = zeros(length(h), length(y), length(s))

    for s_id in s
        for y_id in y
        # Van det heijde
        # 1 : Selecting days by fitting the duration curve    
        # From Selecting Representative Days for Capturing the Implications of Integrating Intermittent Renewables in Generation Expansion Planning Problems by Kris poncelet et al.

            days_selected, weights = get_days(N_days, N_bins, ω, y_id, s_id; time_limit = time_limit[1])

            days[:, y_id, s_id] = days_selected
        #######################
        ### Recompose data to plot and compare the load curves ######
        #####################

                if display_res
                    fig, axs = PyPlot.subplots(4,1, figsize=(9, 3), sharey=false, constrained_layout = true)
                    data_reshape = []

                    push!(data_reshape, reshape(ω.demands[1].power[:, y_id, s_id], (24,365)))
                    push!(data_reshape, reshape(ω.demands[2].power[:, y_id, s_id], (24,365)))
                    push!(data_reshape, reshape(ω.generations[1].power[:, y_id, s_id], (24,365)))

                    data = []

                    push!(data, ω.demands[1].power[:, y_id, s_id])
                    push!(data, ω.demands[2].power[:, y_id, s_id])
                    push!(data, ω.generations[1].power[:, y_id, s_id])

                    labels = ["Load E", "Load H", "Generation"]


                    for j in 1:3

                        val = []
                        for i in 1:length(days_selected)
                            val = vcat(val, repeat(data_reshape[j][:,days_selected[i]], outer = weights[i]))
                        end
                    
                        RP_DC = reverse(sort(val))
                    
                        OG_DC = reverse(sort(data[j]))
                        
                                
                        axs[j].plot(RP_DC, label="Bins = $N_bins, Days = $N_days")
                        axs[j].plot(OG_DC, label = "OG")

                        axs[j].set_title(string("Duration curve : ", labels[j]))
                        axs[j].set_xlabel("Hours",fontsize = 14)
                        axs[j].set_ylabel(j!=3 ? "Power [kW]" : "Power [p.u]",fontsize = 16)
                        
                        
                    end

                    color_names = collect(keys(matplotlib.colors.XKCD_COLORS))[6:2:end]
                    count_start = 1
                    id_x = []
                    for i in 1:length(weights)
                        push!(id_x, (count_start,weights[i]))
                        axs[4].annotate(days_selected[i], (count_start-3 + weights[i]/2, 0.5))
                        count_start += weights[i]
                    end
                    axs[4].broken_barh(id_x , (0, 1),
                            facecolors=color_names[days_selected])

                    
                            
                    legend()
                end
                

        # 2 : Assign real days to representative days to reconstruct the temporality    
        # From Representative days selection for district energy system optimisation: a solar district heating system with seasonal storage
        # Contruct a MIQP model to fit the original data curves by constructing a new one with representative days

                load_E, load_H, gen, sequence_repr = get_profil_and_sequence(days_selected, weights, ω, y_id, s_id; display_res = display_res, time_limit = time_limit[2])
                    

                loads_E[:,y_id,s_id] = load_E
                loads_H[:,y_id,s_id] = load_H

                PVs[:,y_id,s_id] = gen


                sequence[:, y_id, s_id] = sequence_repr

                
                for i in 1:N_days
                    for j in 1:24
                        index_hour[(i-1)*24+j, y_id, s_id] = (days_selected[i]-1) * 24 + j
                    end
                end

        end
    end

        
    for (k, a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            demands[k] = (t = ω.demands[k].t[index_hour], power = ω.demands[k].power[index_hour])
            demands_reconstructed[k] = (t = ω.demands[k].t, power = loads_E)
        elseif a.carrier isa Heat
            demands[k] = (t = ω.demands[k].t[index_hour], power = ω.demands[k].power[index_hour])
            demands_reconstructed[k] = (t = ω.demands[k].t, power = loads_H)
        end
    end    

    for (k, a) in enumerate(mg.generations)
        if a isa Solar
            generations[k] = (t = ω.generations[k].t[index_hour], power = ω.generations[k].power[index_hour], cost = ω.generations[k].cost[y, s])
            generations_reconstructed[k] = (t = ω.generations[k].t, power = PVs, cost = ω.generations[k].cost[y, s])       
        end
    end


    # Grids
    for (k, a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            grids[k] = (cost_in = ω.grids[k].cost_in[index_hour], cost_out = ω.grids[k].cost_out[index_hour], cost_exceed = zeros(length(y),length(s)) .+ 10) #TODO this price should come from the scenarios
        end
    end

    Scenarios_reconstructed = Scenarios(demands_reconstructed, generations_reconstructed,  ω.storages, ω.converters, ω.grids)
   

    return MiniScenarios(demands, generations,  ω.storages, ω.converters, grids, days, sequence, Scenarios_reconstructed)

end


"""
    function Scenarios(mg::Microgrid, d::Dict{})

Constructor function for creating a new `Scenarios` instance based on a given `Microgrid` and a `Dict` containing scenario data.

# Arguments
- `mg::Microgrid`: A Microgrid instance.
- `d::Dict{}`: A dictionary containing scenario data.

# Returns
- `Scenarios`: A Scenarios instance with the specified data.

## Example

```julia
microgrid = ...
scenario_data = ...

scenarios = Scenarios(microgrid, scenario_data)
```
"""
function MiniScenarios(mg::Microgrid, ω::Scenarios, N_days::Int64, y; N_bins = 20, times_limit = [100,200], display_res = true)
    h, s = 1:mg.parameters.nh, 1:mg.parameters.ns
    T, O, I = Array{DateTime,3}, Array{Float64, 3}, Array{Float64, 2}, Array{Int64, 3}


    demands = Vector{NamedTuple}(undef, length(mg.demands))
    generations = Vector{NamedTuple}(undef, length(mg.generations))
    storages = Vector{NamedTuple}(undef, length(mg.storages))
    converters = Vector{NamedTuple}(undef, length(mg.converters))
    grids = Vector{NamedTuple}(undef, length(mg.grids))

    demands_reconstructed = Vector{NamedTuple}(undef, length(mg.demands))
    generations_reconstructed = Vector{NamedTuple}(undef, length(mg.generations))
    storages_reconstructed = Vector{NamedTuple}(undef, length(mg.storages))
    converters_reconstructed = Vector{NamedTuple}(undef, length(mg.converters))
    grids_reconstructed = Vector{NamedTuple}(undef, length(mg.grids))

    days = Int.(zeros(N_days, mg.parameters.ns))
    sequence = Int.(zeros(365*y, mg.parameters.ns))

    index_hour = Int.(zeros(24*N_days, mg.parameters.ns))

    data = []
    constructed_res = zeros(length(h) * y, length(s))
    loads_E = zeros(length(h) * y, length(s))
    loads_H = zeros(length(h) * y, length(s))
    PVs = zeros(length(h) * y, length(s))

    for s_id in s
       
        # Van det heijde
        # 1 : Selecting days by fitting the duration curve    
        # From Selecting Representative Days for Capturing the Implications of Integrating Intermittent Renewables in Generation Expansion Planning Problems by Kris poncelet et al.

            days_selected, weights = get_days_multi_year(N_days, N_bins, ω, y, s_id; time_limit = times_limit[1])

            days[:, s_id] = days_selected
        #######################
        ### Recompose data to plot and compare the load curves ######
        #####################

                if display_res
                  
                    data_reshape = []

                    for demand in ω.demands
                        push!(data_reshape, reshape(vec(demand.power[:, 1:y, s]),(24,365*y)))
                    end
                    for generation in ω.generations
                        push!(data_reshape, reshape(vec(generation.power[:, 1:y, s]),(24,365*y)))
                    end
                    # push!(data_reshape, reshape(vec(ω.demands[1].power[:, : , s_id]), (24,365*y)))
                    # push!(data_reshape, reshape(vec(ω.demands[2].power[:, :, s_id]), (24,365*y)))
                    # push!(data_reshape, reshape(vec(ω.generations[1].power[:, :, s_id]), (24,365*y)))

                    data = []
                    labels = []
                    units = []

                    for (k,demand) in enumerate(ω.demands)
                        push!(data, vec(demand.power[:, 1:y, s]))
                        push!(labels, string("demand : ", typeof(mg.demands[k].carrier)))
                        push!(units, string( "Power [kWh]"))

                    end
                    for (k,generation) in enumerate(ω.generations)
                        push!(data, vec(generation.power[:, 1:y, s]))
                        push!(labels, string("generation : ", typeof(mg.generations[k])))
                        push!(units, string("Power [p.u]"))
                    end
                    # push!(data, vec(ω.demands[1].power[:, :, s_id]))
                    # push!(data, vec(ω.demands[2].power[:, :, s_id]))
                    # push!(data, vec(ω.generations[1].power[:, :, s_id]))
                    
                    fig, axs = PyPlot.subplots(length(data)+1,1, figsize=(9, 3), sharey=false, constrained_layout = true)
                    fig.set_size_inches( 1920 / fig.dpi, 1080/ fig.dpi)
                
                    for j in 1:length(data)

                        val = []
                        for i in 1:length(days_selected)
                            val = vcat(val, repeat(data_reshape[j][:,days_selected[i]], outer = weights[i]))
                        end
                    
                        RP_DC = reverse(sort(val))
                    
                        OG_DC = reverse(sort(data[j]))
                        
                                
                        axs[j].plot(RP_DC, label="Bins = $N_bins, Days = $N_days")
                        axs[j].plot(OG_DC, label = "OG")

                        axs[j].set_title(string("Duration curve : ", labels[j]))
                        axs[j].set_xlabel("Hours",fontsize = 14)
                        axs[j].set_ylabel(units[j], fontsize = 16)
                        
                        
                    end

                    color = Seaborn.color_palette("viridis", as_cmap =true)
                    #color_names = collect(keys(matplotlib.colors.XKCD_COLORS))[1:end]
                    count_start = 1
                    id_x = []
                    for i in 1:length(weights)
                        push!(id_x, (count_start,weights[i]))
                        axs[length(data)+1].annotate(days_selected[i], (count_start -0.5 + weights[i]/2, 0.5))
                        count_start += weights[i]
                    end
                    axs[length(data)+1].broken_barh(id_x , (0, 1),
                            facecolors=color.(days_selected./(365*y))) #color_names[days_selected])

                    
                    tight_layout()

                    legend()
                end
                

        # 2 : Assign real days to representative days to reconstruct the temporality    
        # From Representative days selection for district energy system optimisation: a solar district heating system with seasonal storage
        # Contruct a MIQP model to fit the original data curves by constructing a new one with representative days

                constructed_res, sequence_repr = get_profil_and_sequence_multi_year(days_selected, weights, ω, y, s_id, mg; display_res = display_res, time_limit = times_limit[2])
                    

                #loads_E[:,:,s_id] = load_E
                #loads_H[:,:,s_id] = load_H

               # PVs[:,:,s_id] = gen


                sequence[:, :, s_id] = sequence_repr

                
                for i in 1:N_days
                    for j in 1:24
                        index_hour[(i-1)*24+j, s_id] = (days_selected[i]-1) * 24 + j
                    end
                end

    end

        
    for (k, a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            demands[k] = (t = ω.demands[k].t[index_hour], power = ω.demands[k].power[index_hour])
            demands_reconstructed[k] = (t = ω.demands[k].t, power = constructed_res[k])
        elseif a.carrier isa Heat
            demands[k] = (t = ω.demands[k].t[index_hour], power = ω.demands[k].power[index_hour])
            demands_reconstructed[k] = (t = ω.demands[k].t, power = loads_H)
        end
    end    

    for (k, a) in enumerate(mg.generations)
        if a isa Solar
            generations[k] = (t = ω.generations[k].t[index_hour], power = ω.generations[k].power[index_hour], cost = ω.generations[k].cost)
            generations_reconstructed[k] = (t = ω.generations[k].t, power = constructed_res[k+length(mg.demands)], cost = ω.generations[k].cost)       
        end
    end


    # Grids
    for (k, a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            grids[k] = (cost_in = ω.grids[k].cost_in[index_hour], cost_out = ω.grids[k].cost_out[index_hour], cost_exceed = zeros(length(y),length(s)) .+ 10) #TODO this price should come from the scenarios
        end
    end

    Scenarios_reconstructed = Scenarios(demands_reconstructed, generations_reconstructed,  ω.storages, ω.converters, ω.grids)
   

    return MiniScenarios_my(demands, generations,  ω.storages, ω.converters, grids, days, sequence, Scenarios_reconstructed)

end





function mean_over_days(nweeks::Int, ns::Int, data)
    power = zeros(24*7*nweeks, ns)
    for s in 1:ns
        for i in 1:nweeks
            indexes = Int(1+(i-1)*(8760/nweeks)):Int(i*8760/nweeks)
            data_power_tmp = data.power[indexes,2,s]
            data_time_tmp = data.t[indexes,2,s]

            for id_day in 1:7
                for id_hour in 0:23
                    id_final = ((i-1)*168)+ (id_day-1)*24 +1 + id_hour
                    power[id_final,s] = mean(data_power_tmp[dayofweek.(data_time_tmp) .== id_day .&& hour.(data_time_tmp) .== id_hour])
                
                end
            end
        end
    end
    return power
end

"""
    function Scenarios(mg::Microgrid, d::Dict{}; same_year = false, seed = [])

Constructor function for creating a new `Scenarios` instance based on a given `Microgrid` and a `Dict` containing scenario data. It allows for repetitive years for longer scenarios.

# Arguments
- `mg::Microgrid`: A Microgrid instance.
- `d::Dict{}`: A dictionary containing scenario data.

# Keyword Arguments
- `same_year::Bool=false`: If `true`, the function repeats the same year for all years in the microgrid.
- `seed::Array=[]`: An array specifying the seed for selecting the scenario number and offer reproductivity.

# Returns
- `Scenarios`: A Scenarios instance with the specified data.

## Example

```julia
microgrid = ...
scenario_data = ...

scenarios = Scenarios(microgrid, scenario_data; same_year = true, seed = [1, 2, 3])
```
"""

function Scenarios(mg::Microgrid, d::Dict{}, nweeks::Int64; seed = []) # repeat make every year the same, seed decide with year to use.


    # Utils to simplify the writting
    h, y, s = 1:(nweeks*7*24), 1:mg.parameters.ny, 1:mg.parameters.ns
    T, O, I = Array{DateTime,3}, Array{Float64, 3}, Array{Float64, 2}

    power_format = ones(length(h), 2 , length(s) )
    # Initialize
    demands = Vector{NamedTuple{(:t, :power),Tuple{T,O}}}(undef, length(mg.demands))
    generations = Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}(undef, length(mg.generations))
    storages = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.storages))
    converters = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.converters))
    grids = Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}(undef, length(mg.grids))
    # Demands
    for (k, a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            demands[k] = (t = d["ld_E"].t[h, y, s], power = power_format .* mean_over_days(nweeks, mg.parameters.ns, d["ld_E"]))
        elseif a.carrier isa Heat
            demands[k] = (t = d["ld_H"].t[h, y, s], power = power_format .* mean_over_days(nweeks, mg.parameters.ns, d["ld_H"]))
        end
    end
    # Generation
    for (k, a) in enumerate(mg.generations)
        if a isa Solar
            generations[k] = (t = d["pv"].t[h, y, s], power = power_format .* mean_over_days(nweeks, mg.parameters.ns, d["pv"]), cost = d["pv"].cost[y, s])
        end
    end
    # Storages
    for (k, a) in enumerate(mg.storages)

        if typeof(a) <: AbstractLiion
            storages[k] = (cost = d["liion"].cost[y, s],)
        elseif a isa ThermalStorage
            storages[k] = (cost = d["tes"].cost[y, s],)
        elseif a isa H2Tank
            storages[k] = (cost = d["h2tank"].cost[y, s],)
        end
    end
    # Converters
    for (k, a) in enumerate(mg.converters)
        if typeof(a) <: AbstractElectrolyzer
            converters[k] = (cost = d["elyz"].cost[y, s],)
        elseif typeof(a) <: AbstractFuelCell
            converters[k] = (cost = d["fc"].cost[y, s],)
        elseif a isa Heater
            converters[k] = (cost = d["heater"].cost[y, s],)
        end
    end
    # Grids
    for (k, a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            grids[k] = (cost_in = d["grid_Elec"].cost_in[h, y, s], cost_out = d["grid_Elec"].cost_out[h, y, s], cost_exceed = zeros(length(y),length(s)) .+ 10) #TODO this price should come from the scenarios
        end
    end

    return Scenarios(demands, generations, storages, converters, grids)
end


namedtuple(x) = x
namedtuple(d::Dict) = (; (Symbol(k) => namedtuple(v) for (k,v) in d)...)



"""
    function Scenarios(mg::Microgrid, d::Dict{}, same_year::Bool; seed = [])

Constructor function for creating a new `Scenarios` instance based on a given `Microgrid` and a `Dict` containing scenario data. It allows for repetitive years for longer scenarios.

# Arguments
- `mg::Microgrid`: A Microgrid instance.
- `d::Dict{}`: A dictionary containing scenario data.
- `same_year::Bool` : If `true`, the function repeats the same year for all years in the microgrid.
- `seed::Array=[]`: An array specifying the seed for selecting the scenario number and offer reproductivity.

# Returns
- `Scenarios`: A Scenarios instance with the specified data.

## Example

```julia
microgrid = ...
scenario_data = ...

scenarios = Scenarios(microgrid, scenario_data, true; seed = [1, 2, 3])
```
"""
function Scenarios(mg::Microgrid, d::Dict{}, same_year; seed = []) # repeat make every year the same, seed decide with year to use.
    # Utils to simplify the writting
    nh = mg.parameters.nh
    ny = mg.parameters.ny
    ns = mg.parameters.ns

    h, y, s = 1:nh, 1:1, 1:ns

    # Initialize
    demands = []
    generations = []
    storages = []
    converters = []
    grids = []
    # Demands

    if !isempty(seed) && same_year # Si rep et seed
        @assert(length(seed) == ns, "When you use seed with rep, the seeds for the scenarios must be of length ns")
    elseif !isempty(seed) && !same_year # Si seed mais pas rep
        @assert(size(seed) == (ny,ns), "When you use seed without rep, the seeds for the scenarios must be of size (ny, ns)")
    end

    #Indices of one year scenarios to use.
    r = convert.(Int64, ones(ny,ns))
    for sc in 1:ns
        for ye in 1:ny
            if !isempty(seed) 
                if same_year 
                    r[ye,sc] = seed[sc]
                else
                    r[ye,sc] = seed[ye,sc]
                end
            else
                if same_year && ny == 1
                    r[ye,sc] = convert(Int64, floor(rand() * ns)+1)
                else 
                    r[ye,sc] = convert(Int64, floor(rand() * ns)+1)
                end
            end
        end
    end


    for a in mg.demands
        if a.carrier isa Electricity
            push!(demands, Dict("t" => repeat(d["ld_E"]["t"][h, y, s], outer = (1,ny,1)), "power" => compose3D(d["ld_E"]["power"], mg, r)))
        elseif a.carrier isa Heat
            push!(demands, Dict("t" => repeat(d["ld_H"]["t"][h, y, s], outer = (1,ny,1)), "power" => compose3D(d["ld_H"]["power"], mg, r)))
        end
    end
    # Generation
    for a in mg.generations
        if a isa Solar
            push!(generations, Dict("t" => repeat(d["pv"]["t"][h, y, s], outer = (1,ny,1)), "power" => compose3D(d["pv"]["power"], mg, r), "cost" =>  compose2D(d["pv"]["cost"], mg, r)))
        end
    end
    # Storages
    for a in mg.storages

        if typeof(a) <: AbstractLiion
            key = "liion"
             #push!(storages, Dict("cost" => compose2D(d["liion"]["cost"], mg, r)))
        elseif a isa ThermalStorage
            key = "tes"
            #push!(storages, Dict("cost" => compose2D(d["tes"]["cost"], mg, r)))
        elseif a isa H2Tank
            key = "h2tank"
            #push!(storages, Dict("cost" => compose2D(d["h2tank"]["cost"], mg, r)))
        elseif a isa Barrage
            key = "dam"
            #push!(storages, Dict("cost" => compose2D(d["dam"]["cost"], mg, r),  "irradiance" => compose3D(d["dam"]["irradiance"], mg, r),  "rain" => compose3D(d["dam"]["rain"], mg, r)))
        end

        tmp = Dict()
        for (k,v) in d[key]
            tmp[k] = length(size(v)) == 2 ? compose2D(v, mg, r) : compose3D(v, mg, r)
        end

        push!(storages, tmp)

    end

    # Converters
    for a in mg.converters
        if a isa Electrolyzer
            push!(converters, Dict("cost" => compose2D(d["elyz"]["cost"], mg, r)))
        elseif a isa FuelCell
            push!(converters, Dict("cost" => compose2D(d["fc"]["cost"], mg, r)))
        elseif a isa Heater
            push!(converters, Dict("cost" => compose2D(d["heater"]["cost"], mg, r)))
        end
    end
    # Grids
    for a in mg.grids
        if a.carrier isa Electricity
            push!(grids, Dict("cost_in" => compose3D(d["grid_Elec"]["cost_in"], mg, r), "cost_out" => compose3D(d["grid_Elec"]["cost_out"], mg, r), "cost_exceed" => compose2D(d["grid_Elec"]["cost_exceed"], mg, r)))
        elseif a.carrier isa Heat
            push!(grids, Dict("cost_in" => compose3D(d["grid_Heat"]["cost_in"], mg, r), "cost_out" => compose3D(d["grid_Heat"]["cost_out"], mg, r), "cost_exceed" => compose2D(d["grid_Heat"]["cost_exceed"], mg, r)))
        elseif a.carrier isa Hydrogen
            push!(grids, Dict("cost_in" => compose3D(d["grid_Hydrogen"]["cost_in"], mg, r), "cost_out" => compose3D(d["grid_Hydrogen"]["cost_out"], mg, r), "cost_exceed" => compose2D(d["grid_Hydrogen"]["cost_exceed"], mg, r)))

        end
    end
    

    return Scenarios([namedtuple(x) for x in demands],
        [namedtuple(x) for x in generations],
        [namedtuple(x) for x in storages],
        [namedtuple(x) for x in converters],
        [namedtuple(x) for x in grids]
    )
end




function compose3D(array, mg::Microgrid, seq::Matrix{Int64})


    nh = mg.parameters.nh
    ny = mg.parameters.ny
    ns = mg.parameters.ns

    if typeof(array[1,1,1]) == DateTime
        return repeat(array, outer = (1,ny,1))
    end
    
    result = zeros(nh, ny, ns)

    for s in 1:ns
        for y in 1:ny
            result[:,y,s] = array[:,1,seq[y,s]] # Get the  whole second year of a random scenario and plug in the selected year
        end
    end

    return result

end

function compose2D(array, mg::Microgrid, seq::Matrix{Int64})
    ny = mg.parameters.ny
    ns = mg.parameters.ns

    
    result = zeros(ny, ns)

    for s in 1:ns
        for y in 1:ny
            result[y,s] = array[1,seq[y,s]] # Get the  whole second year of a random scenario and plug in the selected year
        end
    end

    return result
end



function Scenarios(mg::Microgrid, d::Dict, k::Int64, algo::String; weeks = false)
    h, y, s = 1:mg.parameters.nh, 1:mg.parameters.ny, 1:1
    T, O, I = Array{DateTime,3}, Array{Float64, 3}, Array{Float64, 2}


    demands = Vector{NamedTuple{(:t, :power),Tuple{T,O}}}(undef, length(mg.demands))
    generations = Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}(undef, length(mg.generations))
    storages = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.storages))
    converters = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.converters))
    grids = Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}(undef, length(mg.grids))

    max_ld = maximum(d["ld_E"].power[h, 2, 1])
    max_pv = maximum(d["pv"].power[h, 2, 1])

    

    #Normilized data
    data_ld_norm = d["ld_E"].power[h, 2, 1] ./ max_ld
    data_pv_norm = d["pv"].power[h, 2, 1]./ max_pv

    
    if weeks 
        hours = 24*7
        periods = 52
    else
        hours = 24
        periods = 365
    end
   

    data_agg = zeros(periods,hours*2)
    for j in 1:periods
        interval = ((j-1)*hours+1):(j*hours)
        data_agg[j,1:hours] = data_ld_norm[interval]
        data_agg[j,(hours+1):(2*hours)] = data_pv_norm[interval]
    end

  
    

    if algo == "kmedoids"
        dist = pairwise(Euclidean(), data_agg, dims = 1 )


        clusters = kmedoids(dist, k)




        index_hour = Int.(zeros(hours*k))
        for i in 1:k
            for j in 1:hours
                index_hour[(i-1)*hours+j] = (clusters.medoids[i]-1) * hours + j
            end
        end

     
        sort!(index_hour)

        for (k, a) in enumerate(mg.demands)
            if a.carrier isa Electricity
                demands[k] = (t = d["ld_E"].t[index_hour, y, s], power =  d["ld_E"].power[index_hour, y, s])
            elseif a.carrier isa Heat
                demands[k] = (t = d["ld_H"].t[index_hour, y, s], power = d["ld_H"].power[index_hour, y, s])
            end
        end    

        for (k, a) in enumerate(mg.generations)
            if a isa Solar
                generations[k] = (t = d["pv"].t[index_hour, y, s], power = d["pv"].power[index_hour, y, s], cost = d["pv"].cost[y, s])
            end
        end

       
        # Grids
        for (k, a) in enumerate(mg.grids)
            if a.carrier isa Electricity
                grids[k] = (cost_in = d["grid_Elec"].cost_in[index_hour, y, s], cost_out = d["grid_Elec"].cost_out[index_hour, y, s], cost_exceed = zeros(length(y),length(s)) .+ 10) #TODO this price should come from the scenarios
            end
        end


    elseif algo == "kmeans"


        #Random.seed!(1)
        #clusters = kmeans(transpose(data_agg), k, init=[rand((1, 365)) for i in 1:k])
        clusters = kmeans(transpose(data_agg), k)
        


        load = hcat(ones((hours*k),1,1) .* vec(clusters.centers[1:hours,:]) .* max_ld, ones((hours*k),1,1) .* vec(clusters.centers[1:hours,:]) .* max_ld)
        gen =  hcat(ones((hours*k),1,1) .* vec(clusters.centers[(hours+1):(2*hours),:]) .* max_pv, ones((hours*k),1,1) .* vec(clusters.centers[(hours+1):(2*hours),:]) .* max_pv)
        
        for (i, a) in enumerate(mg.demands)
            if a.carrier isa Electricity
                demands[i] = (t = d["ld_E"].t[1:(hours*k), y, s], power = load)
            elseif a.carrier isa Heat
                demands[i] = (t = d["ld_H"].t[1:(hours*k), y, s], power = load.*0)
            end
        end    

        for (i, a) in enumerate(mg.generations)
            if a isa Solar
                generations[i] = (t = d["pv"].t[1:(hours*k), y, s], power = gen, cost = d["pv"].cost[y, s])
            end
        end

    
        # Grids
        for (i, a) in enumerate(mg.grids)
            if a.carrier isa Electricity
                grids[i] = (cost_in = d["grid_Elec"].cost_in[1:(hours*k), y, s], cost_out = d["grid_Elec"].cost_out[1:(hours*k), y, s], cost_exceed = zeros(length(y),length(s)) .+ 10) #TODO this price should come from the scenarios
            end
        end


    
    else 
        print("not an accepted clustering algorithm")
    end


    for (k, a) in enumerate(mg.storages)

        if typeof(a) <: AbstractLiion
            storages[k] = (cost = d["liion"].cost[y, s],)
        elseif a isa ThermalStorage
            storages[k] = (cost = d["tes"].cost[y, s],)
        elseif a isa H2Tank
            storages[k] = (cost = d["h2tank"].cost[y, s],)
        end
    end
    # Converters
    for (k, a) in enumerate(mg.converters)
        if typeof(a) <: AbstractElectrolyzer
            converters[k] = (cost = d["elyz"].cost[y, s],)
        elseif typeof(a) <: AbstractFuelCell
            converters[k] = (cost = d["fc"].cost[y, s],)
        elseif a isa Heater
            converters[k] = (cost = d["heater"].cost[y, s],)
        end
    end

    return Scenarios(demands, generations, storages, converters, grids),  clusters.assignments

end















