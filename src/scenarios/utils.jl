#=
    Utility functions
=#
# Concatenate scenario
function Base.cat(ω1::Scenarios, ω2::Scenarios; dims::Int64 = 2)
    @assert 1 < dims < 4 "concatenation along dims 2 and 3 only !"
    # Initialize
    demands, generations, storages, converters, grids = similar(ω1.demands), similar(ω1.generations), similar(ω1.storages), similar(ω1.converters), similar(ω1.grids)
    # Demands
    for k in 1:length(ω1.demands)
        demands[k] = (t = cat(ω1.demands[k].t, ω2.demands[k].t, dims=dims), power = cat(ω1.demands[k].power,  ω2.demands[k].power, dims=dims))
    end
    # Generations
    for k in 1:length(ω1.generations)
        generations[k] = (t = cat(ω1.generations[k].t, ω2.generations[k].t, dims=dims), power =  cat(ω1.generations[k].power, ω2.generations[k].power, dims=dims), cost =  cat(ω1.generations[k].cost, ω2.generations[k].cost, dims=dims-1))
    end
    # Storages
    for k in 1:length(ω1.storages)
        storages[k] = (cost =  cat(ω1.storages[k].cost, ω2.storages[k].cost, dims=dims-1),)
    end
    # Converters
    for k in 1:length(ω1.converters)
        converters[k] = (cost =  cat(ω1.converters[k].cost, ω2.converters[k].cost, dims=dims-1),)
    end
    # Grids
    for k in 1:length(ω1.grids)
        grids[k] = (cost_in = cat(ω1.grids[k].cost_in, ω2.grids[k].cost_in, dims=dims), cost_out =  cat(ω1.grids[k].cost_out, ω2.grids[k].cost_out, dims=dims))
    end

    return Scenarios(demands, generations, storages, converters, grids)
end
# Repeat scenario
function Base.repeat(ω::Scenarios, nh::Int64, ny::Int64, ns::Int64)
    # Initialize
    demands, generations, storages, converters, grids = similar(ω.demands), similar(ω.generations), similar(ω.storages), similar(ω.converters), similar(ω.grids)
    # Demands
    for k in 1:length(ω.demands)
        demands[k] = (t = repeat(ω.demands[k].t, nh, ny, ns), power = repeat(ω.demands[k].power, nh, ny, ns))
    end
    # Generations
    for k in 1:length(ω.generations)
        generations[k] = (t = repeat(ω.generations[k].t, nh, ny, ns), power =  repeat(ω.generations[k].power, nh, ny, ns), cost =  repeat(ω.generations[k].cost, ny, ns))
    end
    # Storages
    for k in 1:length(ω.storages)
        storages[k] = (cost =  repeat(ω.storages[k].cost, ny, ns),)
    end
    # Converters
    for k in 1:length(ω.converters)
        converters[k] = (cost =  repeat(ω.converters[k].cost, ny, ns),)
    end
    # Grids
    for k in 1:length(ω.grids)
        grids[k] = (cost_in =  repeat(ω.grids[k].cost_in, nh, ny, ns), cost_out =  repeat(ω.grids[k].cost_out, nh, ny, ns))
    end

    return Scenarios(demands, generations, storages, converters, grids)
end
# Reshape scenario
function Base.reshape(ω::Scenarios, nh::Int64, ny::Int64, ns::Int64)
    # Initialize
    demands, generations, storages, converters, grids = similar(ω.demands), similar(ω.generations), similar(ω.storages), similar(ω.converters), similar(ω.grids)
    # Demands
    for k in 1:length(ω.demands)
        demands[k] = (t = reshape(ω.demands[k].t, nh, ny, ns), power = reshape(ω.demands[k].power, nh, ny, ns))
    end
    # Generations
    for k in 1:length(ω.generations)
        generations[k] = (t = reshape(ω.generations[k].t, nh, ny, ns), power =  reshape(ω.generations[k].power, nh, ny, ns), cost =  reshape(ω.generations[k].cost, ny, ns))
    end
    # Storages
    for k in 1:length(ω.storages)
        storages[k] = (cost =  reshape(ω.storages[k].cost, ny, ns),)
    end
    # Converters
    for k in 1:length(ω.converters)
        converters[k] = (cost =  reshape(ω.converters[k].cost, ny, ns),)
    end
    # Grids
    for k in 1:length(ω.grids)
        grids[k] = (cost_in =  reshape(ω.grids[k].cost_in, nh, ny, ns), cost_out =  reshape(ω.grids[k].cost_out, nh, ny, ns), cost_exceed = reshape(ω.grids[k].cost_exceed, ny, ns))
    end

    return Scenarios(demands, generations, storages, converters, grids)
end
# True if t is a weeken day
isweekend(t::Union{DateTime, Array{DateTime}}) = (Dates.dayname.(t) .== "Saturday") .| (Dates.dayname.(t) .== "Sunday")
# Chose the right markovchain as a function of t
chose(generator::MarkovGenerator, t::DateTime) = isweekend(t) ? generator.markovchains.wkd : generator.markovchains.wk
# Generate a yearly profile from typical days clustered data
generate(data_td::Array{Float64,2}, assignments::Array{Int64,1}) = reshape(data_td[:, assignments, :], size(data_td, 1) * size(assignments, 1))
# Normalization
min_max_normalization(x::AbstractArray) = maximum(x) == minimum(x) ? x ./ maximum(x) : (x .- minimum(x)) ./ (maximum(x) .- minimum(x))





function get_profil_and_sequence(days::Vector{Int64}, weights::Vector{Int64}, ω::Scenarios, y::Int64, s::Int64; display_res = false, time_limit = 300)
    
    data_reshape = []

    max_ld_E = maximum(ω.demands[1].power[:, y, s])
    max_ld_H = maximum(ω.demands[2].power[:, y, s])
    max_ld_PV = maximum(ω.generations[1].power[:, y, s])
    
    push!(data_reshape, reshape(ω.demands[1].power[:, y, s], (24,365)) ./ max_ld_E)
    push!(data_reshape, reshape(ω.demands[2].power[:, y, s], (24,365)) ./ max_ld_H)
    push!(data_reshape, reshape(ω.generations[1].power[:, y, s], (24,365)) ./ max_ld_PV)
    
    total_energy = sum(ω.demands[1].power[:, y, s]) + sum(ω.demands[2].power[:, y, s]) + sum(ω.generations[1].power[:, y, s])
    weight_energy = [sum(ω.demands[1].power[:, y, s]), sum(ω.demands[2].power[:, y, s]), sum(ω.generations[1].power[:, y, s])] / total_energy

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
    @objective(m2, Min, sum((error[data_id, d, h]^2) .* weight_energy for data_id in 1:length(data_reshape) for d in 1:365 for  h in 1:24))


    optimize!(m2)

    
    sequence = [findfirst( x -> x > 0, Int64.(round.(value.(m2[:assignments])[i,:]))) for i in 1:365]
    load_result_E = vec(transpose(value.(m2[:constructed_data][1,:,:]))) .*max_ld_E
    load_result_H = vec(transpose(value.(m2[:constructed_data][2,:,:]))) .*max_ld_H
    gen_result = vec(transpose(value.(m2[:constructed_data][3,:,:]))) .*max_ld_PV

    if display_res 
    
        fig, axs = PyPlot.subplots(3,1, figsize=(9, 3), sharey=true)

        axs[1].plot(vec(transpose(value.(m2[:error])[1,:,:])).^2)
        axs[1].set_title("Load_E")
        axs[1].set_xlabel("Days",fontsize = 16)
        axs[1].set_ylabel("Squared Error",fontsize = 16)

        axs[1].plot(vec(transpose(value.(m2[:error])[2,:,:])).^2)
        axs[1].set_title("Load_H")
        axs[1].set_xlabel("Days",fontsize = 16)
        axs[1].set_ylabel("Squared Error",fontsize = 16)


        axs[2].plot(vec(transpose(value.(m2[:error])[3,:,:])).^2)
        axs[2].set_title("Generation")
        axs[2].set_xlabel("Days",fontsize = 16)
        axs[2].set_ylabel("Squared Error",fontsize = 16)


        fig, axs = PyPlot.subplots(4,1, figsize=(9, 3), sharey=false)

        axs[1].plot(vec(data_reshape[1]).* max_ld_E, label = "OG")
        axs[1].plot(vec(transpose(value.(m2[:constructed_data][1,:,:]))) .* max_ld_E)
        axs[1].set_title("Load Profil Elec")
        axs[1].set_ylabel("Power [kW]",fontsize = 16)

        axs[2].plot(vec(data_reshape[2]).* max_ld_H, label = "OG")
        axs[2].plot(vec(transpose(value.(m2[:constructed_data][2,:,:]))) .* max_ld_H)
        axs[2].set_title("Load Profil Heat")
        axs[2].set_ylabel("Power [kW]",fontsize = 16)


        axs[3].plot(vec(data_reshape[3]).* max_ld_PV)
        axs[3].plot(vec(transpose(value.(m2[:constructed_data][3,:,:]))) .* max_ld_PV)
        axs[3].set_title("Generation Profil")
        axs[3].set_xlabel("Hours",fontsize = 16)
        axs[3].set_ylabel("Power [p.u]",fontsize = 16)


        x_id = []
        color_id = []
        color_names = collect(keys(matplotlib.colors.XKCD_COLORS))[6:2:end]
    
        for day in 1:365
            push!(x_id, (((day-1)*1), 1))
            push!(color_id, color_names[sequence[day]])
        end
        axs[4].broken_barh(x_id, (0, 1),
                    facecolors=color_id)

        legend()
    end



    
    return  load_result_E, load_result_H, gen_result, sequence


end



function get_days(N_days, N_bins, ω, y ,s; time_limit = 0)


    data = []

    push!(data, ω.demands[1].power[:, y, s])
    push!(data, ω.demands[2].power[:, y, s])
    push!(data, ω.generations[1].power[:, y, s])

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


