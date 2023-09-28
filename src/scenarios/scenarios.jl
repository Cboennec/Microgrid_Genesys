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
mutable struct Scenarios{T, O, I} <: AbstractScenarios
    demands::Vector{NamedTuple{(:t, :power),Tuple{T,O}}}
    generations::Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}
    storages::Vector{NamedTuple{(:cost,), Tuple{I}}}
    converters::Vector{NamedTuple{(:cost,), Tuple{I}}}
    grids::Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}
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
mutable struct MiniScenarios{T, O, I} <: AbstractScenarios
    demands::Vector{NamedTuple{(:t, :power),Tuple{T,O}}}
    generations::Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}
    storages::Vector{NamedTuple{(:cost,), Tuple{I}}}
    converters::Vector{NamedTuple{(:cost,), Tuple{I}}}
    grids::Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}} 
    days::Array{Int64, 3}
    sequence::Array{Int64, 3}
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
        if a isa Electrolyzer
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

    days = Int.(zeros(N_days, mg.parameters.ny, mg.parameters.ns))
    sequence = Int.(zeros(365, mg.parameters.ny, mg.parameters.ns))

    index_hour = Int.(zeros(24*N_days, mg.parameters.ny, mg.parameters.ns))


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
                    fig, axs = PyPlot.subplots(3,1, figsize=(9, 3), sharey=false, constrained_layout = true)
                    data_reshape = []

                    push!(data_reshape, reshape(ω.demands[1].power[:, y_id, s_id], (24,365)))
                    push!(data_reshape, reshape(ω.generations[1].power[:, y_id, s_id], (24,365)))

                    data = []

                    push!(data, ω.demands[1].power[:, y_id, s_id])
                    push!(data, ω.generations[1].power[:, y_id, s_id])


                    for j in 1:2

                        val = []
                        for i in 1:length(days_selected)
                            val = vcat(val, repeat(data_reshape[j][:,days_selected[i]], outer = weights[i]))
                        end
                    
                        RP_DC = reverse(sort(val))
                    
                        OG_DC = reverse(sort(data[j]))
                        
                                
                        axs[j].plot(RP_DC, label="Bins = $N_bins, Days = $N_days")
                        axs[j].plot(OG_DC, label = "OG")
                        axs[j].set_title(j==1 ? "Duration curve : Load" : "Duration curve : Generation" )
                        axs[j].set_xlabel("Hours",fontsize = 14)
                        axs[j].set_ylabel(j==1 ? "Power [kW]" : "Power [p.u]",fontsize = 16)
                        
                        
                    end

                    color_names = collect(keys(matplotlib.colors.XKCD_COLORS))[6:2:end]
                    count_start = 1
                    id_x = []
                    for i in 1:length(weights)
                        push!(id_x, (count_start,weights[i]))
                        axs[3].annotate(days_selected[i], (count_start-3 + weights[i]/2, 0.5))
                        count_start += weights[i]
                    end
                    axs[3].broken_barh(id_x , (0, 1),
                            facecolors=color_names[days_selected])

                    
                            
                    legend()
                end
                

        # 2 : Assign real days to representative days to reconstruct the temporality    
        # From Representative days selection for district energy system optimisation: a solar district heating system with seasonal storage
        # Contruct a MIQP model to fit the original data curves by constructing a new one with representative days

                load, gen, sequence_repr = get_profil_and_sequence(days_selected, weights, ω, y_id, s_id; display_res = display_res, time_limit = time_limit[2])
                    
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
                demands[k] = (t = ω.demands[1].t[index_hour], power = ω.demands[1].power[index_hour])
            elseif a.carrier isa Heat
                demands[k] = (t = ω.demands[2].t[index_hour], power = ω.demands[2].power[index_hour])
            end
        end    

        for (k, a) in enumerate(mg.generations)
            if a isa Solar
                generations[k] = (t = ω.generations[1].t[index_hour], power = ω.generations[1].power[index_hour], cost = ω.generations[1].cost[y, s])
            end
        end

    
        # Grids
        for (k, a) in enumerate(mg.grids)
            if a.carrier isa Electricity
                grids[k] = (cost_in = ω.grids[1].cost_in[index_hour], cost_out = ω.grids[1].cost_out[index_hour], cost_exceed = zeros(length(y),length(s)) .+ 10) #TODO this price should come from the scenarios
            end
        end


        storages = ω.storages
        converters = ω.converters
   

    return MiniScenarios(demands, generations, storages, converters, grids, days, sequence)

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
        if a isa Electrolyzer
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


function Scenarios(mg::Microgrid, d::Dict{}; same_year = false, seed = []) # repeat make every year the same, seed decide with year to use.
    # Utils to simplify the writting


    h, y, s = 1:mg.parameters.nh, 2:2, 1:mg.parameters.ns
    T, O, I = Array{DateTime,3}, Array{Float64, 3}, Array{Float64, 2}

    rep_time = convert(Int64,  mg.parameters.ny)
    # Initialize
    demands = Vector{NamedTuple{(:t, :power),Tuple{T,O}}}(undef, length(mg.demands))
    generations = Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}(undef, length(mg.generations))
    storages = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.storages))
    converters = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.converters))
    grids = Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}(undef, length(mg.grids))
    # Demands
    for (k, a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            demands[k] = (t = repeat(d["ld_E"].t[h, y, s], outer = (1,rep_time,1)), power = compose(d["ld_E"].power, rep_time, mg, 3; rep = same_year, s_num = seed))
        elseif a.carrier isa Heat
            demands[k] = (t = repeat(d["ld_H"].t[h, y, s], outer = (1,rep_time,1)), power = compose(d["ld_H"].power, rep_time, mg, 3; rep = same_year, s_num = seed))
        end
    end
    # Generation
    for (k, a) in enumerate(mg.generations)
        if a isa Solar
            #generations[k] = (t = d["pv"].t[h, y, s], power = d["pv"].power[h, y, s], cost = d["pv"].cost[y, s])
            generations[k] = (t = repeat(d["pv"].t[h, y, s], outer = (1,rep_time,1)), power = compose( d["pv"].power, rep_time, mg, 3; rep = same_year, s_num = seed), cost = compose(d["pv"].cost, rep_time, mg, 2;  rep = same_year, s_num = seed))
        end
    end
    # Storages
    for (k, a) in enumerate(mg.storages)

        if typeof(a) <: AbstractLiion
        #    storages[k] = (cost = d["liion"].cost[y, s],)
             storages[k] = (cost = compose(d["liion"].cost, rep_time, mg, 2; rep = same_year, s_num = seed),)
        elseif a isa ThermalStorage
        #    storages[k] = (cost = d["tes"].cost[y, s],)
            storages[k] = (cost = compose(d["tes"].cost, rep_time, mg, 2; rep = same_year, s_num = seed),)
        elseif a isa H2Tank
        #    storages[k] = (cost = d["h2tank"].cost[y, s],)
            storages[k] = (cost = compose(d["h2tank"].cost, rep_time, mg, 2; rep = same_year, s_num = seed),)
        end
    end
    # Converters
    for (k, a) in enumerate(mg.converters)
        if a isa Electrolyzer
            #converters[k] = (cost = d["elyz"].cost[y, s],)
            converters[k] = (cost = compose(d["elyz"].cost, rep_time, mg, 2; rep = same_year, s_num = seed),)
        elseif typeof(a) <: AbstractFuelCell
            #converters[k] = (cost = d["fc"].cost[y, s],)
            converters[k] = (cost = compose(d["fc"].cost, rep_time, mg, 2; rep = same_year, s_num = seed),)

        elseif a isa Heater
            #converters[k] = (cost = d["heater"].cost[y, s],)
            converters[k] = (cost = compose(d["heater"].cost, rep_time, mg, 2; rep = same_year, s_num = seed),)
        end
    end
    # Grids
    for (k, a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            #grids[k] = (cost_in = d["grid"].cost_in[h, y, s], cost_out = d["grid"].cost_out[h, y, s])
            grids[k] = (cost_in = compose(d["grid_Elec"].cost_in, rep_time, mg, 3; rep = same_year, s_num = seed), cost_out = compose(d["grid_Elec"].cost_out, rep_time, mg, 3; rep = same_year, s_num = seed), cost_exceed = zeros( mg.parameters.ny,  mg.parameters.ns) .+ 10.2)#TODO this price should come from the scenarios
        elseif a.carrier isa Hydrogen
            grids[k] = (cost_in = compose(d["grid_Hydrogen"].cost_in, rep_time, mg, 3; rep = same_year, s_num = seed), cost_out = compose(d["grid_Hydrogen"].cost_out, rep_time, mg, 3; rep = same_year, s_num = seed), cost_exceed = zeros( mg.parameters.ny,  mg.parameters.ns) )
        elseif a.carrier isa Heat
            print("ERROR Heat grid not coded yet")
        end
    end

    return Scenarios(demands, generations, storages, converters, grids)
end


#Compose the data for a longer scenario based on existing scenario of 1 year
#The dim parameter defines the number of dimension on which this data is represented
#optionnal
#The rep parameter defines wether or not every year have to be the same
#s_num defines the id of the scenarios we are going to use, very useful for reproductivity of the results
# If no seed is provided, the scenario IDs will  be randomly selected.
function compose(array, rep_time::Int64, mg::Microgrid, dim::Int64; rep = false, s_num = [] )
    nh = mg.parameters.nh
    ny = mg.parameters.ny
    ns = mg.parameters.ns

    hours = 1:nh
    years = 1:ny
    scenarios = 1:ns

    r = 0

    if dim == 3
        result = repeat(array[hours, 2:2, scenarios], outer = (1,rep_time,1)) # instantiate an array of the right size.

        for s in scenarios
            if !isempty(s_num)
                r =  s_num[s]
            else
                r = convert(Int64, floor(rand() * 1000)+1)
            end

            for y in years
                if !rep #On refait un tirage
                    r = convert(Int64, floor(rand() * 1000)+1)
                end
                result[:,y,s] = array[:,2,r] # Get the  whole second year of a random scenario and plug in the selected year
            end
        end
    elseif dim == 2
        result = repeat(array[2:2, scenarios], outer = (rep_time,1))

        for s in scenarios
            if !isempty(s_num)
                r =  s_num[s]
            else
                r = convert(Int64, floor(rand() * 1000)+1)
            end

            for y in years
                if !rep #On refait un tirage
                    r = convert(Int64, floor(rand() * 1000)+1)
                end
                result[y,s] = array[2,r] #Get the  whole second year of a random scenario
            end
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
        if a isa Electrolyzer
            converters[k] = (cost = d["elyz"].cost[y, s],)
        elseif typeof(a) <: AbstractFuelCell
            converters[k] = (cost = d["fc"].cost[y, s],)
        elseif a isa Heater
            converters[k] = (cost = d["heater"].cost[y, s],)
        end
    end

    return Scenarios(demands, generations, storages, converters, grids),  clusters.assignments

end















