# Anticipative controller

mutable struct AnticipativeOptions
    solver

    AnticipativeOptions(; solver = Gurobi) = new(solver)
end

mutable struct Anticipative <: AbstractController
    options::AnticipativeOptions
    generations::Dict
    storages::Dict
    converters::Dict
    decisions::NamedTuple

    Anticipative(; options = AnticipativeOptions(),
                   generations = Dict(),
                   storages = Dict(),
                   converters = Dict()) =
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
    fix_investment_decisions!(m, mg, [x for x in values(controller.generations)], [x for x in values(controller.storages)], [x.surface * x.N_cell for x in values(controller.converters)])
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
    # Objective
    opex = compute_opex(m, mg, ω, nh, ns)
    @objective(m, Min, opex[1])

    return m
end


# Model with the addition of a penalization for activating and de-activating the fuelcell 
# If used, this penalization should be weighted as the share of SoH lost.

function build_model_test(mg::Microgrid, controller::Anticipative, ω::Scenarios; )
    # Sets
    nh, ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 3)


    # Initialize
    m = Model(controller.options.solver.Optimizer)
    
    set_optimizer_attribute(m, "TimeLimit", 20)
    #set_optimizer_attribute(m,"CPX_PARAM_SCRIND", 0)
    # Add investment variables
    add_investment_decisions!(m, mg.generations)
    add_investment_decisions!(m, mg.storages)
    add_investment_decisions!(m, mg.converters)
    # Fix their values
    fix_investment_decisions!(m, mg, controller.generations, controller.storages, controller.converters)
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
    # Objective
    opex = compute_opex(m, mg, ω, nh, ns)

    add_FC_decisions!(m, nh, ns)
    add_FC_constraints!(m, mg, nh, ns)
    penalization = compute_penalization(m, nh, ns)

    @objective(m, Min, opex[1]+penalization[1])

    return m
end



function build_model(mg::Microgrid, controller::Anticipative, ω::MiniScenarios, y::Int64, s::Int64)

   
    nh, ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 3)

    # Initialize
    m = Model(controller.options.solver.Optimizer)
    set_optimizer_attribute(m, "TimeLimit", 20)

    #set_optimizer_attribute(m,"CPX_PARAM_SCRIND", 0)
    # Add investment variables
    add_investment_decisions!(m, mg.generations)
    add_investment_decisions!(m, mg.storages)
    add_investment_decisions!(m, mg.converters)
    # Fix their values
    fix_investment_decisions!(m, mg, controller.generations, controller.storages, controller.converters)
    # Add decision variables
    add_operation_decisions!(m, mg.storages, nh, ns)
    add_operation_decisions!(m, mg.converters, nh, ns)
    add_operation_decisions!(m, mg.grids, nh, ns)
    # Add technical constraints
    add_SoC_base!(m, mg.storages, ns)


    add_technical_constraints_mini!(m, mg.storages, mg.parameters.Δh, nh, ns)
    add_technical_constraints!(m, mg.converters, nh, ns)
    add_technical_constraints!(m, mg.grids, nh, ns)
    # Add periodicity constraint
    #add_periodicity_constraints!(m, mg.storages, ns)
    # Add power balance constraints
    add_power_balance!(m, mg, ω, Electricity, nh, ns)
    add_power_balance!(m, mg, ω, Heat, nh, ns)
    add_power_balance!(m, mg, ω, Hydrogen, nh, ns)

    add_Continuity_SoC_constraints_mini!(m, mg.storages, nh, ns, ω.sequence[:,y,s])
    add_periodicity_constraints_mini!(m, mg.storages, ns, ω.sequence[:,y,s])

    # Objective
    opex = compute_opex_mini(m, mg, ω, nh, ns, ω.sequence[:,y,s])
    @objective(m, Min, opex[1])

    return m
end 


function build_model(mg::Microgrid, controller::Anticipative, ω::MiniScenarios_my, s::Int64; time_limit = 3600)

    id_dict = get_id_dict(mg)    
    nh, ns = size(ω.demands[1].power, 1) * size(ω.demands[1].power, 2), size(ω.demands[1].power, 3)
    n_days = size(ω.demands[1].power, 2) * 365
    # Initialize
    m = Model(controller.options.solver.Optimizer)
    set_optimizer_attribute(m, "TimeLimit", time_limit)

    #set_optimizer_attribute(m,"CPX_PARAM_SCRIND", 0)
    # Add investment variables
    add_investment_decisions!(m, mg.generations)
    add_investment_decisions!(m, mg.storages)
    add_investment_decisions!(m, mg.converters)
    # Fix their values
    fix_investment_decisions!(m, mg, controller.generations, controller.storages, controller.converters, id_dict)
    # Add decision variables
    add_operation_decisions!(m, mg.storages, nh, ns)
    add_operation_decisions!(m, mg.converters, nh, ns)
    add_operation_decisions!(m, mg.grids, nh, ns)
    # Add technical constraints
    add_SoC_base!(m, mg.storages, ns, n_days)


    add_technical_constraints_mini!(m, mg.storages, mg.parameters.Δh, nh, ns)
    add_technical_constraints!(m, mg.converters, nh, ns)
    add_technical_constraints!(m, mg.grids, nh, ns)
    # Add periodicity constraint
    #add_periodicity_constraints!(m, mg.storages, ns)
    # Add power balance constraints
    add_power_balance_my!(m, mg, ω, Electricity, nh, ns)
    add_power_balance_my!(m, mg, ω, Heat, nh, ns)
    add_power_balance_my!(m, mg, ω, Hydrogen, nh, ns)

    add_Continuity_SoC_constraints_mini!(m, mg.storages, nh, ns, ω.sequence[:,s])
    add_periodicity_constraints_mini!(m, mg.storages, ns, ω.sequence[:,s])

    # Objective
    opex = compute_opex_mini(m, mg, ω, n_days, ns, ω.sequence[:,s])
    @objective(m, Min, opex[1])

    return m
end 


### Offline
function initialize_controller!(mg::Microgrid, controller::Anticipative, ω::MiniScenarios)
    # Preallocate

    preallocate!(mg, controller)

    model_return = []

    for y in 1:mg.parameters.ny, s in 1:mg.parameters.ns
        # Scenario reduction
       

        # Build model
        model = build_model(mg, controller, ω, y, s)
      
        # Optimize
        JuMP.optimize!(model)

        push!(model_return, model)
        #println(objective_value(model))
        
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
            elseif typeof(a) <: AbstractElectrolyzer
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][h_seq,1,k])
            elseif typeof(a) <: AbstractFuelCell
                controller.decisions.converters[k][:,y,s] .= value.(model[:p_c][h_seq,1,k])
            end
        end
    
       
    end

    return controller, model_return
end


function initialize_controller!(mg::Microgrid, controller::Anticipative, ω::MiniScenarios_my; time_limit = 3600)
    # Preallocate

    preallocate!(mg, controller)

    model_return = []

    for s in 1:mg.parameters.ns
        # Scenario reduction
       

        # Build model
        model = build_model(mg, controller, ω, s; time_limit)
      
        # Optimize
        JuMP.optimize!(model)

        push!(model_return, model)
        #println(objective_value(model))
        
        legend()

       
      
        for y in 1:mg.parameters.ny

            h_seq = []
            for d in (1+(y-1)*365):(365 + (y-1)*365)
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
                    controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][h_seq,y,k])
                elseif typeof(a) <: AbstractElectrolyzer
                    controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][h_seq,y,k])
                elseif typeof(a) <: AbstractFuelCell
                    controller.decisions.converters[k][:,y,s] .= value.(model[:p_c][h_seq,y,k])
                end
            end
        end
    
       
    end

    return controller, model_return
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
            elseif typeof(a) <: AbstractElectrolyzer
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][:,1,k])
            elseif typeof(a) <: AbstractFuelCell
                controller.decisions.converters[k][:,y,s] .= value.(model[:p_c][:,1,k])
            end
        end
    
       
    end

    return controller
end



### Offline
function initialize_controller_test!(mg::Microgrid, controller::Anticipative, ω::Scenarios)
    # Preallocate

    preallocate!(mg, controller)

    models = []

    for y in 1:mg.parameters.ny, s in 1:mg.parameters.ns
        # Scenario reduction
        if mg.parameters.ns == 1  
            ω_reduced = ω
        else
            ω_reduced, _ = reduce(ManualReducer(h = 1:mg.parameters.nh, y = y:y, s = s:s), ω)
        end
        # Build model


        model = build_model_test(mg, controller, ω_reduced)
        # Optimize
        optimize!(model)

        
                # Assign controller values
        for k in 1:length(mg.storages)
            controller.decisions.storages[k][:,y,s] .= value.(model[:p_dch][:,1,k] .- model[:p_ch][:,1,k])
        end
        for (k,a) in enumerate(mg.converters)
            if a isa Heater
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][:,1,k])
            elseif typeof(a) <: AbstractElectrolyzer
                controller.decisions.converters[k][:,y,s] .= .- value.(model[:p_c][:,1,k])
            elseif typeof(a) <: AbstractFuelCell
                controller.decisions.converters[k][:,y,s] .= value.(model[:p_c][:,1,k])
            end
        end
    
       push!(models, model)
    end

    return controller, models
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
            elseif typeof(a) <: AbstractElectrolyzer
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
