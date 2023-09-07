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








function Scenarios_repr(mg::Microgrid, d::Dict, N_days::Int64; N_bins = 20, time_limit = [100,200], display_res = true)
    h, y, s = 1:mg.parameters.nh, 1:mg.parameters.ny, 1:1
    T, O, I = Array{DateTime,3}, Array{Float64, 3}, Array{Float64, 2}


    demands = Vector{NamedTuple{(:t, :power),Tuple{T,O}}}(undef, length(mg.demands))
    generations = Vector{NamedTuple{(:t, :power, :cost), Tuple{T, O, I}}}(undef, length(mg.generations))
    storages = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.storages))
    converters = Vector{NamedTuple{(:cost,), Tuple{I}}}(undef, length(mg.converters))
    grids = Vector{NamedTuple{(:cost_in, :cost_out, :cost_exceed), Tuple{O, O, I}}}(undef, length(mg.grids))


# Van det heijde
# 1 : Selecting days by fitting the duration curve    
# From Selecting Representative Days for Capturing the Implications of Integrating Intermittent Renewables in Generation Expansion Planning Problems by Kris poncelet et al.

    days, weights = get_days(N_days, N_bins, d; time_limit = time_limit[1])

#######################
### Recompose data to plot and compare the load curves ######
#####################

        if display_res
            fig, axs = PyPlot.subplots(3,1, figsize=(9, 3), sharey=false)
            data_reshape = []

            push!(data_reshape, reshape(d["ld_E"].power[:, 2, 1], (24,365)))
            push!(data_reshape, reshape(d["pv"].power[:, 2, 1], (24,365)))

            data = []

            push!(data, d["ld_E"].power[:, 2, 1])
            push!(data, d["pv"].power[:, 2, 1])


            for j in 1:2

                val = []
                for i in 1:length(days)
                    val = vcat(val, repeat(data_reshape[j][:,days[i]], outer = weights[i]))
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
                axs[3].annotate(days[i], (count_start-3 + weights[i]/2, 0.5))
                count_start += weights[i]
            end
            axs[3].broken_barh(id_x , (0, 1),
                       facecolors=color_names[days])

            
                       
            legend()
        end
        

# 2 : Assign real days to representative days to reconstruct the temporality    
# From Representative days selection for district energy system optimisation: a solar district heating system with seasonal storage
# Contruct a MIQP model to fit the original data curves by constructing a new one with representative days

        load, gen, sequence = get_profil_and_sequence(days, weights, d; display_res = display_res, time_limit = time_limit[2])


       

     
            

        index_hour = Int.(zeros(24*N_days))
        
        for i in 1:N_days
            for j in 1:24
                index_hour[(i-1)*24+j] = (days[i]-1) * 24 + j
            end
        end


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

    return Scenarios(demands, generations, storages, converters, grids), days, sequence

end













function get_profil_and_sequence(days, weights, data_raw; display_res = false, time_limit = 300)
    
    data_reshape = []

    max_ld_E = maximum(data_raw["ld_E"].power[:, 2, 1])
    max_ld_PV = maximum(data_raw["pv"].power[:, 2, 1])
    
    push!(data_reshape, reshape(data_raw["ld_E"].power[:, 2, 1], (24,365)) ./ max_ld_E)
    push!(data_reshape, reshape(data_raw["pv"].power[:, 2, 1], (24,365)) ./ max_ld_PV)
    


    m2 = Model(Gurobi.Optimizer)
    set_optimizer_attribute(m2, "TimeLimit", time_limit)
    
    #Which day is assigned to which representative
    @variable(m2, assignments[1:365, 1:length(days)], Bin)

    #Each representative represent itself
    for (i,r) in enumerate(days)
        fix(m2[:assignments][r,i], 1)
    end

    #Each day is represented by one day
    @constraint(m2, [d in 1:365], sum(assignments[d,r] for r in 1:length(days)) == 1)
    # Each representative represents a number of day equal to its weight
    @constraint(m2, [r in 1:length(days)], sum(assignments[d,r] for d in 1:365) == weights[r])

    #The constructed profil
    @variable(m2, constructed_data[1:length(data_reshape), 1:365, 1:24])
    #Assign values to the constructed profil
    @constraint(m2, [data_id in 1:length(data_reshape), d in 1:365, h in 1:24], constructed_data[data_id,d,h] == sum(assignments[d,r] * data_reshape[data_id][h,days[r]] for r in 1:length(days)))

    @variable(m2, error[1:length(data_reshape), 1:365, 1:24])
    @constraint(m2, [data_id in 1:length(data_reshape), d in 1:365, h in 1:24], error[data_id, d, h] == (constructed_data[data_id,d,h] - data_reshape[data_id][h,d]))

    #Minimize the squared error
    @objective(m2, Min, sum(error[data_id, d, h]^2 for data_id in 1:length(data_reshape) for d in 1:365 for  h in 1:24))


    optimize!(m2)

    
    sequence = [findfirst( x -> x > 0, Int64.(round.(value.(m2[:assignments])[i,:]))) for i in 1:365]
    load_result = vec(transpose(value.(m2[:constructed_data][1,:,:]))) .*max_ld_E
    gen_result = vec(transpose(value.(m2[:constructed_data][2,:,:]))) .*max_ld_PV

    if display_res 
    
        fig, axs = PyPlot.subplots(1,2, figsize=(9, 3), sharey=true)

        axs[1].plot(vec(transpose(value.(m2[:error])[1,:,:])).^2)
        axs[1].set_title("Load")
        axs[1].set_xlabel("Days",fontsize = 16)
        axs[1].set_ylabel("Squared Error",fontsize = 16)


        axs[2].plot(vec(transpose(value.(m2[:error])[2,:,:])).^2)
        axs[2].set_title("Generation")
        axs[2].set_xlabel("Days",fontsize = 16)
        axs[2].set_ylabel("Squared Error",fontsize = 16)


        fig, axs = PyPlot.subplots(3,1, figsize=(9, 3), sharey=false)

        axs[1].plot(vec(data_reshape[1]).* max_ld_E, label = "OG")
        axs[1].plot(vec(transpose(value.(m2[:constructed_data][1,:,:]))) .* max_ld_E)
        axs[1].set_title("Load Profil")
        axs[1].set_xlabel("Hours",fontsize = 16)
        axs[1].set_ylabel("Power [kW]",fontsize = 16)

        axs[2].plot(vec(data_reshape[2]).* max_ld_PV)
        axs[2].plot(vec(transpose(value.(m2[:constructed_data][2,:,:]))) .* max_ld_PV)
        axs[2].set_title("Generation Profil")
        axs[2].set_xlabel("Hours",fontsize = 16)
        axs[2].set_ylabel("Power [p.u]",fontsize = 16)


        x_id = []
        color_id = []
        color_names = collect(keys(matplotlib.colors.XKCD_COLORS))[6:2:end]
        color_id = color_names[days]
    
        for day in 1:365
            push!(x_id, (((day-1)*1), 1))
            push!(color_id, color_names[sequence[day]])
        end
        axs[3].broken_barh(x_id, (0, 1),
                    facecolors=color_id)

        legend()
    end



    
    return  load_result, gen_result, sequence


end



function get_days(N_days, N_bins, data_raw; time_limit = 0)


    data = []

    push!(data, data_raw["ld_E"].power[:, 2, 1])
    push!(data, data_raw["pv"].power[:, 2, 1])

    N_metric = length(data) # Elec curve, Solar curve

    
    # #######################
    ### Define values of L and A (see Poncelet et al. P.5 second column) ######
    #####################
    L = zeros(N_metric, N_bins)
    A = zeros(N_metric, N_bins, 365)

    for i in 1:N_metric
        min_data = minimum(data[i])
        max_data = maximum(data[i])

        bin_step = (max_data-min_data)/N_bins

        OG_DC = reverse(sort(data[i]))



        for j in 1:N_bins
            #PyPlot.scatter(findfirst(OG_DC .<  ((j-1)*bin_step)+min_data),((j-1)*bin_step)+min_data)

            L[i,j] = sum(OG_DC .>= ((j-1)*bin_step)+min_data) / 8760

            for k in 1:365
                A[i,j,k] = sum(data[i][((k-1)*24+1):(k*24)] .>= ((j-1)*bin_step)+min_data) / 24
            end
        end
    end

    


#######################
### Optimize to find the best set of days and their weights ######
#####################

    m = Model(Gurobi.Optimizer)

    if time_limit > 0
        set_optimizer_attribute(m, "TimeLimit", time_limit)
    end


    @variable(m, weight_day[1:365] >= 0, Int)
    @variable(m, day[1:365], Bin)


    @constraint(m, sum(m[:weight_day][d] for d in 1:365) == 365)
    @constraint(m, [d in 1:365], weight_day[d] <= 365 * day[d])
    @constraint(m, [d in 1:365], weight_day[d] >= day[d])


    @constraint(m, sum(day[d] for d in 1:365) == N_days)

    @variable(m, errors[1:N_metric, 1:N_bins])


    @constraint(m, [metric in 1:N_metric, b in 1:N_bins], errors[metric,b] >= sum(weight_day[d] * 1/365 * A[metric,b,d] for d in 1:365) - L[metric,b])
    @constraint(m, [metric in 1:N_metric, b in 1:N_bins], errors[metric,b] >= -(sum(weight_day[d] * 1/365 * A[metric,b,d] for d in 1:365) - L[metric,b]))

    @objective(m, Min, sum(errors[metric, b] for metric in 1:N_metric for b in 1:N_bins))

    optimize!(m)

    days_id = findall( x -> x > 0, Int64.(value.(m[:day])))
    return days_id, Int64.(value.(m[:weight_day]))[days_id]
            
end


