#=
    This file includes all the funtions needed to update the operation
    and investment informations
 =#

function update_operation_informations!(h::Int64, y::Int64, s::Int64, mg::Microgrid, ω::AbstractScenarios)
    # Demands
    for (k, a) in enumerate(mg.demands)
        @inbounds a.timestamp[h,y,s] = ω.demands[k].t[h,y,s]
        @inbounds a.carrier.power[h,y,s] = ω.demands[k].power[h,y,s]
    end
    # Generations
    for (k, a) in enumerate(mg.generations)
        @inbounds a.timestamp[h,y,s] = ω.generations[k].t[h,y,s]
        @inbounds a.carrier.power[h,y,s] = a.powerMax[y,s] * ω.generations[k].power[h,y,s]
    end

end

function update_investment_informations!(y::Int64, s::Int64, mg::Microgrid, ω::AbstractScenarios)

    # Generations
    for (k, a) in enumerate(mg.generations)
        @inbounds a.cost[y,s] = ω.generations[k].cost[y,s]
    end
    # Storages
    for (k, a) in enumerate(mg.storages)
        @inbounds a.cost[y,s] = ω.storages[k].cost[y,s]
    end
    # Converters
    for (k, a) in enumerate(mg.converters)
        @inbounds a.cost[y,s] = ω.converters[k].cost[y,s]
    end

end


function update_grid_cost_informations!(y::Int64, s::Int64, mg::Microgrid, ω::AbstractScenarios)
    # Grids - We assume the price of electricity is known over the year
    for (k, a) in enumerate(mg.grids)
        @inbounds a.cost_in[1:end,y,s] = ω.grids[k].cost_in[1:end,y,s]
        @inbounds a.cost_out[1:end,y,s] = ω.grids[k].cost_out[1:end,y,s]
        @inbounds a.cost_exceed[y,s] = ω.grids[k].cost_exceed[y,s]
    end

end
