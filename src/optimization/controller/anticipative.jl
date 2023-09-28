# Anticipative controller

mutable struct AnticipativeOptions
    solver

    AnticipativeOptions(; solver = Gurobi) = new(solver)
end

mutable struct Anticipative <: AbstractController
    options::AnticipativeOptions
    generations::Vector{Float64}
    storages::Vector{Float64}
    converters::Vector{Float64}
    decisions::NamedTuple

    Anticipative(; options = AnticipativeOptions(),
                   generations = [0.],
                   storages = [0.],
                   converters = [0.]) =
                   new(options, generations, storages, converters)
end

### Models
function build_model(mg::Microgrid, controller::Anticipative, ω::Scenarios; representative = false, assignments=[])
    # Sets
    nh, ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 3)

    factor = zeros(nh)
    if length(assignments) > 0
        for i in 1:Int(nh/24)
            factor[((i-1)*24+1):(i*24)] .= count(==(i), assignments)
        end
    end
    # Initialize
    m = Model(controller.options.solver.Optimizer)
    #set_optimizer_attribute(m,"CPX_PARAM_SCRIND", 0)
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
    if representative
        add_technical_constraints!(m, mg.storages, mg.parameters.Δh, nh, ns, representative, factor)
    else
        add_technical_constraints!(m, mg.storages, mg.parameters.Δh, nh, ns, false)
    end
    add_technical_constraints!(m, mg.converters, nh, ns)
    add_technical_constraints!(m, mg.grids, nh, ns)
    # Add periodicity constraint
    add_periodicity_constraints!(m, mg.storages, ns)
    # Add power balance constraints
    add_power_balance!(m, mg, ω, Electricity, nh, ns)
    add_power_balance!(m, mg, ω, Heat, nh, ns)
    add_power_balance!(m, mg, ω, Hydrogen, nh, ns)
    # Objective
    opex = compute_opex(m, mg, ω, nh, ns)
    @objective(m, Min, opex[1])
    return m
end



function build_model(mg::Microgrid, controller::Anticipative, ω::MiniScenarios, y::Int64, s::Int64)

   
    nh, ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 3)

    m = Model(controller.options.solver.Optimizer)
    #set_optimizer_attribute(m,"CPX_PARAM_SCRIND", 0)
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

    add_SoC_base!(m, mg.storages, ns)
    # Add technical constraints
    add_technical_constraints!(m, mg.storages, mg.parameters.Δh, nh, ns, true)
    add_technical_constraints!(m, mg.converters, nh, ns)
    add_technical_constraints!(m, mg.grids, nh, ns)
   
    # Add power balance constraints
    add_power_balance!(m, mg, ω, Electricity, nh, ns)
    add_power_balance!(m, mg, ω, Heat, nh, ns)
    add_power_balance!(m, mg, ω, Hydrogen, nh, ns)

    #SoC Constraints
    add_Continuity_SoC_constraints(m, mg.storages, nh, ns, ω.sequence[:,y,s])
    # Add periodicity constraint
    add_periodicity_constraints_mini!(m, mg.storages, ns)

    # Objective
    opex = compute_opex(m, mg, ω, nh, ns)
    @objective(m, Min, opex[1])

    return m
end 


### Offline
function initialize_controller!(mg::Microgrid, controller::Anticipative, ω::MiniScenarios)
    # Preallocate

    preallocate!(mg, controller)

    for y in 1:mg.parameters.ny, s in 1:mg.parameters.ns
        # Scenario reduction
       

        # Build model
        model = build_model(mg, controller, ω, y, s)
      
        # Optimize
        optimize!(model)

        fig, axs = PyPlot.subplots(1,2, figsize=(9, 3), sharey=true)

        axs[1].plot(vec(transpose(value.(model[:soc])[:,y,1])))
        axs[1].set_title("SoC_base battery")
        axs[1].set_xlabel("Days",fontsize = 16)
        axs[1].set_ylabel("SoC",fontsize = 16)


        axs[2].plot(vec(transpose(value.(model[:soc])[:,y,2])))
        axs[2].set_title("SoC_base H2")
        axs[2].set_xlabel("Days",fontsize = 16)
        axs[2].set_ylabel("SoC",fontsize = 16)

        legend()

        h_seq = []
        for d in 1:365
            for h in 1:24
                h_seq = push!(h_seq, (ω.sequence[d]-1) * 24 + h  )
            end
        end
      
                # Assign controller values
        for k in 1:length(mg.storages)
            controller.decisions.storages[k][:,y,s] .= value.(model[:p_dch][h_seq,1,k] .- model[:p_ch][h_seq,1,k])
        end
        for (k,a) in enumerate(mg.converters)
            if a isa Heater
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][h_seq,1,k])
            elseif a isa Electrolyzer
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][h_seq,1,k])
            elseif typeof(a) <: AbstractFuelCell
                controller.decisions.converters[k][:,y,s] .= value.(model[:p_c][h_seq,1,k])
            end
        end
    
       
    end

    return controller
end


### Offline
function initialize_controller!(mg::Microgrid, controller::Anticipative, ω::Scenarios)
    # Preallocate

    preallocate!(mg, controller)

    for y in 1:mg.parameters.ny, s in 1:mg.parameters.ns
        # Scenario reduction
        if mg.parameters.ns == 1  
            ω_reduced = ω
        else
            ω_reduced, _ = reduce(ManualReducer(h = 1:mg.parameters.nh, y = y:y, s = s:s), ω)
        end
        # Build model

        model = build_model(mg, controller, ω_reduced)
        # Optimize
        optimize!(model)

        
                # Assign controller values
        for k in 1:length(mg.storages)
            controller.decisions.storages[k][:,y,s] .= value.(model[:p_dch][:,1,k] .- model[:p_ch][:,1,k])
        end
        for (k,a) in enumerate(mg.converters)
            if a isa Heater
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][:,1,k])
            elseif a isa Electrolyzer
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][:,1,k])
            elseif typeof(a) <: AbstractFuelCell
                controller.decisions.converters[k][:,y,s] .= value.(model[:p_c][:,1,k])
            end
        end
    
       
    end

    return controller
end

### Offline
function initialize_controller!(mg::Microgrid, controller::Anticipative, ω::Scenarios, representative::Bool; assignments = [])
    # Preallocate
    if representative
        ω, _ , sequence = Scenarios_repr(mg, ω, 20)
    end

    preallocate!(mg, controller)

    for y in 1:mg.parameters.ny, s in 1:mg.parameters.ns
        # Scenario reduction
        if mg.parameters.ns == 1  
            ω_reduced = ω
        else
            ω_reduced, _ = reduce(ManualReducer(h = 1:mg.parameters.nh, y = y:y, s = s:s), ω)
        end
        # Build model
        if representative
            model = build_model(mg, controller, ω_reduced)
        else
            model = build_model(mg, controller, ω_reduced, sequence)
        end
        # Optimize
        optimize!(model)

        if representative
            h_seq = []
            for d in 1:365
                for h in 1:24
                    h_seq = push!(h_seq, (sequence[d]-1) * 24 + h  )
                end
            end
        else
            h_seq = 1:8760
        end
                # Assign controller values
        for k in 1:length(mg.storages)
            controller.decisions.storages[k][:,y,s] .= value.(model[:p_dch][h_seq,1,k] .- model[:p_ch][h_seq,1,k])
        end
        for (k,a) in enumerate(mg.converters)
            if a isa Heater
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][h_seq,1,k])
            elseif a isa Electrolyzer
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][h_seq,1,k])
            elseif typeof(a) <: AbstractFuelCell
                controller.decisions.converters[k][:,y,s] .= value.(model[:p_c][h_seq,1,k])
            end
        end
    
       
    end

    return controller
end



### Online
function compute_operation_decisions!(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::Anticipative)
    return controller
end
