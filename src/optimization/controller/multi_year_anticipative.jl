# Anticipative controller

mutable struct Multi_Year_Anticipative_Options
    solver

    Multi_Year_Anticipative_Options(; solver = Cbc) = new(solver)
end

mutable struct Multi_Year_Anticipative <: AbstractController
    options::Multi_Year_Anticipative_Options
    generations::Vector{Float64}
    storages::Vector{Float64}
    converters::Vector{Float64}
    decisions::NamedTuple

    Multi_Year_Anticipative(; options = Multi_Year_Anticipative_Options(),
                   generations = [0.],
                   storages = [0.],
                   converters = [0.]) =
                   new(options, generations, storages, converters)
end


### Models
function build_model_lim_SoH(mg::Microgrid, controller::Multi_Year_Anticipative, designer::AbstractDesigner, ω::Scenarios; lim = 1)
    # Sets
    nh, ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 3)
    # Initialize
    m = Model(controller.options.solver.Optimizer)
    set_optimizer_attribute(m,"CPX_PARAM_SCRIND", 0)
    # Add investment variables
    add_investment_decisions!(m, mg.generations)
    add_investment_decisions!(m, mg.storages)
    add_investment_decisions!(m, mg.converters)
    # Fix their values
    fix_investment_decisions!(m, controller.generations, controller.storages, controller.converters)
    # Add decision variables
    add_operation_decisions!(m, mg.storages, nh, ns)
    add_operation_decisions!(m, mg.converters, nh, ns)
    add_operation_decisions!(m, mg.grids, nh, ns)
    # Add technical constraints
    add_technical_constraints!(m, mg.storages, mg.parameters.Δh, nh, ns)
    add_technical_constraints!(m, mg.converters, nh, ns)
    add_technical_constraints!(m, mg.grids, nh, ns)
    # Add periodicity constraint
    add_periodicity_constraints!(m, mg.storages, ns)
    # Add power balance constraints
    add_power_balance!(m, mg, ω, Electricity, nh, ns)
    add_power_balance!(m, mg, ω, Heat, nh, ns)
    add_power_balance!(m, mg, ω, Hydrogen, nh, ns)

    for a in mg.storages
        if typeof(a) <: AbstractLiion
            E_ex_tot = (2. * a.nCycle * (a.α_soc_max - a.α_soc_min) * designer.storages["liion"]) 
            add_SoH_lim_variables_constraint!(m, k, nh, ns, E_ex_tot, lim)
        end
    end

    # Objective
    opex = compute_opex(m, mg, ω, nh, ns)
    @objective(m, Min, opex[1])
    return m
end


#Get multi year optimal control integrating battery aging
function initialize_controller!(mg::Microgrid, controller::Multi_Year_Anticipative, designer::AbstractDesigner, ω::Scenarios)
    # Preallocate
    preallocate!(mg, controller)


    ###### COMPUTE TRANSITIONS ############
    #Define SoH ranges
    min_Δ = (4/3)/100 # ~= (1 - exp(- 4.14e-10 * 3600)) * 8760 = Δcal
    max_Δ = 1.5*min_Δ
    interval_Δ = min_Δ/4

    #Battery replacement level and battery starting level
    replacement_threshold = mg.storages[1].SoH_threshold 
    SoH_ini = mg.storages[1].soh_ini 

    #Each starting level and each possible range from these levels
    range_from_state = [x for x in (SoH_ini):(-interval_Δ):(replacement_threshold + min_Δ)] 
    ΔSoH_range = min_Δ:interval_Δ:max_Δ

    #Initialize cost and decision data 
    transition_costs = ones(length(range_from_state), length(ΔSoH_range)) .* Inf 
    transition_decisions_storages = []
    transition_decisions_converters = []


    for k in 1:length(mg.storages)
        push!(transition_decisions_storages, zeros(length(range_from_state), length(ΔSoH_range) , mg.parameters.nh, 1, mg.parameters.ns ))
    end
    for k in 1:length(mg.converters)
        push!(transition_decisions_converters, zeros(length(range_from_state), length(ΔSoH_range) , mg.parameters.nh, 1, mg.parameters.ns ))
    end
    

    # TEMPORARY
  
    s=1


   
    #Compute each transition
    for (SoH_ID, SoH_i) in enumerate(range_from_state)
        for (ΔSoH_ID, ΔSoH) in enumerate(ΔSoH_range)
            print((SoH_ID-1)*length(ΔSoH_range)+ΔSoH_ID,"/", length(range_from_state)*length(ΔSoH_range))
            #Without getting over the replacement threshold
            if SoH_i - ΔSoH >= replacement_threshold

                #Initialize start state at SoH_i %
                controller.storages = designer.storages * SoH_i

                Cost = zeros(mg.parameters.ns)

                for s in 1:mg.parameters.ns
                    ω_reduced, _ = reduce(ManualReducer(h = 1:mg.parameters.nh, y = 2:2, s = s:s), ω)
                    #Contruct model for optimal control over 1 year with a maximum usage of the battery inducing ΔSoH degradation
                    model = build_model_lim_SoH(mg, controller, designer, ω_reduced; lim = ΔSoH)
                    optimize!(model)
                    
                    #Get the cost and include some salvage if a part of the battery have been artificially dropped by the discretization of states
                    opex = objective_value(model)
                    effective_ΔSoH = value.(model[:ΔSoH])
                    salvage = (ΔSoH.-effective_ΔSoH) .* (designer.storages["Liion"] .* ω_reduced.storages[1].cost[1,1])
                    
                    Cost[s] = opex[1] - salvage[1]

                    #Store decisions of the optimal control for this transition.
                    for k in 1:length(mg.storages)
                        transition_decisions_storages[k][SoH_ID,ΔSoH_ID,:,1,s] .= value.(model[:p_dch][:,1,k] .- model[:p_ch][:,1,k])
                    end
                    for (k,a) in enumerate(mg.converters)
                        if a isa Heater
                            transition_decisions_converters[k][SoH_ID,ΔSoH_ID,:,1,s] .= .- value.(model[:p_c][:,1,k])
                        elseif a isa Electrolyzer
                            transition_decisions_converters[k][SoH_ID,ΔSoH_ID,:,1,s] .= .- value.(model[:p_c][:,1,k])
                        elseif a isa FuelCell
                            transition_decisions_converters[k][SoH_ID,ΔSoH_ID,:,1,s] .= value.(model[:p_c][:,1,k])
                        end
                    end

                end 

                #Store the final mean cost for the edge
                transition_costs[SoH_ID,ΔSoH_ID] = mean(Cost)

            end
        end
    end

    

    ###### BUILD GRAPH ############
    graph = []
    decision_graph = []
    
    height = Int(round((SoH_ini - replacement_threshold) / interval_Δ)+1)
    max_jump = Int(round(max_Δ/interval_Δ))
    min_jump = Int(round(min_Δ/interval_Δ))
     
    Ny = mg.parameters.ny

    tab_noeuds = zeros(height,Ny+1)
    tab_arcs = zeros(height,Ny+1)

    #Initialization
    tab_noeuds[1,1] = 1
    node_num = 2

    cout_remplacement = (designer.storages["Liion"] .* ω.storages[1].cost[1,1])

    for y in 1:(Ny)
        #On active les noeuds grace aux arcs 

    
        #On active les arcs vers l'année suivante
        for i in 1:height #for each state level
            if tab_noeuds[i,y] >= 1 #if the node is activated
                for j in min_jump:min(max_jump, height-i) # for each step possible
                    tab_arcs[i+j,y+1] = tab_arcs[i+j, y+1] + 1 #Activate the edges
                    if tab_arcs[i+j,y+1] == 1 # When it's the first time a node is reached by an edge
                        tab_noeuds[i+j,y+1] = node_num # We activate the new vertex
                        node_num = node_num+1 #Increment the vertex ID
                    end
                    
                    #We store this new edge in the graph structure
                    push!(graph, (Int(tab_noeuds[i,y]), Int(tab_noeuds[i+j,y+1]), transition_costs[i,j-(min_jump-1)] ))
                    push!(decision_graph, (Int(tab_noeuds[i,y]), Int(tab_noeuds[i+j,y+1]), (i,j-(min_jump-1)) ))

                    #arc de (i,y) à (i+j, y+1) de longueur transition
                end
            end
        end

        # we add the edge for battery replacement
        if tab_arcs[height,y+1] >= 1
            tab_noeuds[1,y+1] = node_num #Go from last state to the first state representing a fresh battery
            node_num = node_num+1

            push!(graph, (Int(tab_noeuds[height,y+1]), Int(tab_noeuds[1,y+1]), cout_remplacement)) # The cost of the edge is the price of a battery

        end

    
    end

    #Salvage final
    for h in 1:height
        if tab_noeuds[h,Ny+1] >= 1
            push!(graph, (Int(tab_noeuds[h,Ny+1]), node_num, - cout_remplacement * ((height-1)-(h-1))/(height-1) ))
        end
    end


    cost, path = bellman_ford(1, graph) 



    #ASSIGN decisions to the controller
    
    replacement_count = 0
    for y in 1:(length(path)-1), s in 1:mg.parameters.ns

        #On trouve l'arc 
        i,j = get_decisions_indices(path[y], path[y+1], decision_graph)
        if (i,j) != (0,0) 

           for k in 1:length(mg.storages)
               controller.decisions.storages[k][:,y-replacement_count,s] .= transition_decisions_storages[k][i,j,:,1,s]
           end
           for k in 1:length(mg.converters)
               controller.decisions.converters[k][:,y-replacement_count,s] .= transition_decisions_converters[k][i,j,:,1,s]
           end
       else 
           replacement_count = replacement_count + 1
       end

    end


    return controller
end

### Online
function compute_operation_decisions!(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::Multi_Year_Anticipative)
    return controller
end



function bellman_ford(s, graph) 
    node_nums = []
    for (u,v,d) in graph
        push!(node_nums, u)
        push!(node_nums, v)
    end
    nb_vertex = length(unique(node_nums))

    short_distances = ones(nb_vertex, nb_vertex).* 10e6
    precedents = zeros(nb_vertex, nb_vertex)

    i=1
    short_distances[i,s] = 0
    stop = false

    while !stop
        short_distances[i+1,:] = short_distances[i,:]
        precedents[i+1,:] = precedents[i,:]
        for (u,v,d) in graph
            if short_distances[i,u] + d < short_distances[i,v] 
                short_distances[i+1,v] = short_distances[i,u] + d
                precedents[i+1,v] = u
            end
        end

        i = i+1
        if short_distances[i,:] == short_distances[i-1,:]
            stop = true
        end
    end


    path_to_final = []
    prec = nb_vertex
    stop = false
    j = (i-1)
    while !stop
        push!(path_to_final, prec)
        if prec == 1
            stop = true
        end
        prec = Int(precedents[j,prec])

        j = j-1
    end

    return short_distances[i,nb_vertex], reverse(path_to_final)
end


function get_decisions_indices(vertex1, vertex2, decision_graph)
    
    for (from, to ,indices) in decision_graph
        if from == vertex1
            if to == vertex2
                return indices[1], indices[2]
            end
        end
    end
    
    return 0,0
end