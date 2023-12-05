#=
    This file includes all the funtions needed to compute the operation
    and investment dynamics
 =#

function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::AbstractController)
    # Converters
    for (k, a) in enumerate(mg.converters)
        compute_operation_dynamics!(h, y, s, a, controller.decisions.converters[k][h,y,s], mg.parameters.Δh)
    end
    # Storage
    for (k, a) in enumerate(mg.storages)
        compute_operation_dynamics!(h, y, s, a, controller.decisions.storages[k][h,y,s], mg.parameters.Δh)
    end
end

function compute_investment_dynamics!(y::Int64, s::Int64, mg::Microgrid, designer::AbstractDesigner)
     # Generations
     for (k, a) in enumerate(mg.generations)
        if a isa Solar
            compute_investment_dynamics!(y, s, a, designer.decisions.generations["PV"][y,s])
        end
     end
     # Storages
     for (k, a) in enumerate(mg.storages)
        if typeof(a) <: AbstractLiion
            compute_investment_dynamics!(y, s, a, designer.decisions.storages["Liion"][y,s])
        elseif a isa H2Tank
            compute_investment_dynamics!(y, s, a, designer.decisions.storages["H2Tank"][y,s])
        end
     end
     # Converters
     for (k, a) in enumerate(mg.converters)

        if a isa FuelCell
            compute_investment_dynamics!(y, s, a, (surface = designer.decisions.converters["FuelCell"].surface[y,s], N_cell = Int(designer.decisions.converters["FuelCell"].N_cell[y,s])) )
        elseif a isa Electrolyzer
            compute_investment_dynamics!(y, s, a, (surface = designer.decisions.converters["Electrolyzer"].surface[y,s], N_cell = Int(designer.decisions.converters["Electrolyzer"].N_cell[y,s])) )
        elseif a isa Heater
            compute_investment_dynamics!(y, s, a, designer.decisions.converters["Heater"][y,s])
        end

     end
end


function initialize_investments!(s::Int64, mg::Microgrid, designer::AbstractDesigner)
    # Generations
    for a in mg.generations
        if a isa Solar
            initialize_investments!(s, a, designer.generations["PV"])
        end
    end

    # Storages
    for a in mg.storages
        if typeof(a) <: AbstractLiion
            initialize_investments!(s, a, designer.storages["Liion"])
        elseif a isa H2Tank
            initialize_investments!(s, a, designer.storages["H2Tank"])
        end
    end
    # Converters
    for a in mg.converters
        if a isa FuelCell
            initialize_investments!(s, a, (surface = designer.converters["FuelCell"].surface, N_cell = designer.converters["FuelCell"].N_cell))
        elseif a isa Electrolyzer
            
            initialize_investments!(s, a, (surface = designer.converters["Electrolyzer"].surface, N_cell = designer.converters["Electrolyzer"].N_cell))
            
        elseif a isa Heater
            initialize_investments!(s, a, designer.converters["Heater"])
        end
    end
end

