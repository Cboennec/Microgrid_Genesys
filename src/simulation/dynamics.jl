#=
    This file includes all the funtions needed to compute the operation
    and investment dynamics
 =#

function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::AbstractController)
    # Converters
    for (k, a) in enumerate(mg.converters)
        @inbounds compute_operation_dynamics!(h, y, s, a, controller.decisions.converters[k][h,y,s], mg.parameters.Δh)
    end
    # Storage
    for (k, a) in enumerate(mg.storages)
        @inbounds compute_operation_dynamics!(h, y, s, a, controller.decisions.storages[k][h,y,s], mg.parameters.Δh)
    end
end

function compute_investment_dynamics!(y::Int64, s::Int64, mg::Microgrid, designer::AbstractDesigner)
     # Generations
     for (k, a) in enumerate(mg.generations)
        compute_investment_dynamics!(y, s, a, designer.decisions.generations[string(typeof(a))][y,s])
     end
     # Storages
     for (k, a) in enumerate(mg.storages)
        compute_investment_dynamics!(y, s, a, designer.decisions.storages[string(typeof(a))][y,s])
     end
     # Converters
     for (k, a) in enumerate(mg.converters)

        if a isa FuelCell || a isa Electrolyzer
           compute_investment_dynamics!(y, s, a, (surface = designer.decisions.converters[string(typeof(a))].surface[y,s], N_cell = Int(designer.decisions.converters[string(typeof(a))].N_cell[y,s])) )
        elseif a isa Heater
            compute_investment_dynamics!(y, s, a, designer.decisions.converters[string(typeof(a))][y,s])
        end
     end
end


function initialize_investments!(s::Int64, mg::Microgrid, designer::AbstractDesigner)
    # Generations
    for a in mg.generations
        initialize_investments!(s, a, designer.generations[string(typeof(a))])
    end

    # Storages
    for a in mg.storages
        initialize_investments!(s, a, designer.storages[string(typeof(a))])
    end
    # Converters
    for a in mg.converters
        if a isa FuelCell || a isa Electrolyzer
            initialize_investments!(s, a, (surface = designer.converters[string(typeof(a))].surface, N_cell = designer.converters[string(typeof(a))].N_cell))            
        elseif a isa Heater
            initialize_investments!(s, a, designer.converters[string(typeof(a))])
        end
    end

    for a in mg.grids
        initialize_investments!(s, a, designer.subscribed_power[string(typeof(a.carrier))])
    end
end

