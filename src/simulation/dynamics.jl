"""
 function compute_operation_dynamics!(h::Int64, y::Int64, s::Int64, mg::Microgrid, controller::AbstractController)

For each convert and storage from mg the operation of the time and scenario index h,y,s the decision from controller are applied.
# Arguments
- `h::Int64`: An hour.
- `y::Int64`: A year.
- `s::Int64`: A scenario.
- `mg::Microgrid`: A microgrid to be operated at time index h,y,s.
- `controller::AbstractController`: A controller storing the decisions for the components to be operated..

# Returns
- nothing

## Example

```julia
    # Compute operation dynamics for each converter and storage in mg
    compute_operation_dynamics!(h, y, s, mg, controller)
```
"""
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

"""
function compute_investment_dynamics!(y::Int64, s::Int64, mg::Microgrid, designer::AbstractDesigner)

For each generation, convertion and storage unit from mg, the design decisions associated to the year y and scenario s are applied.
# Arguments
- `y::Int64`: A year.
- `s::Int64`: A scenario.
- `mg::Microgrid`: A microgrid to be operated at time index h,y,s.
- `designer::AbstractDesigner`: A designer storing the design decisions.

# Returns
- nothing

## Example

```julia
    # Compute decision dynamics for each convertion, storage and generation unit in mg
    compute_investment_dynamics!(y, s, mg, designer)
```
"""
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

