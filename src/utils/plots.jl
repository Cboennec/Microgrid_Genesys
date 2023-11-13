# This file includes all the plot functions for the MG

using Pandas

function powerPlot(energy_carriers::Vector{EnergyCarrier}, mg::Microgrid, y::UnitRange{Int64}, s::UnitRange{Int64}, hours::UnitRange{Int64})
    f = figure("Powers")
    f.subplotpars.hspace = 0.32
    for s_id in s
        for (i, type) in enumerate(energy_carriers)
            i == 1 ? subplot(length(energy_carriers), 1, i, title = string(type)) : subplot(length(energy_carriers), 1, i, sharex = f.axes[1], title = string(type))
            # Demands
            for (k, a) in enumerate(mg.demands)
                if a.carrier isa type
                    Seaborn.plot(hours, -vec(a.carrier.power[:,y,s_id]), label = string("Demand : ",  typeof(a.carrier)))
                end
            end
            # Generations
            for (k, a) in enumerate(mg.generations)
                if a.carrier isa type
                    Seaborn.plot(hours, vec(a.carrier.power[:,y,s_id]), label = string("Generation : ", typeof(a)))
                end
            end
            # Storages
            for (k, a) in enumerate(mg.storages)
                if a.carrier isa type
                    Seaborn.plot(hours, vec(a.carrier.power[:,y,s_id]), label = string("Storage : ", typeof(a)))
                end
            end
            # Converters
            for (k, a) in enumerate(mg.converters)
                for c in a.carrier
                    if c isa type
                        Seaborn.plot(hours, vec(c.power[:,y,s_id]), label = string("Converter : ", typeof(a)))
                    end
                end
            end
            for (k, a) in enumerate(mg.grids)
                if a.carrier isa type
                    Seaborn.plot(hours,  vec(a.carrier.power[:,y,s_id]), label = string("Grids : ", typeof(a)))
                end
            end
            legend()
        end
    end
end

function powerBalancePlot(energy_carriers::Vector{EnergyCarrier}, mg::Microgrid, y::UnitRange{Int64}, s::UnitRange{Int64}, hours::UnitRange{Int64})
    f = figure("Power Balances")
    f.subplotpars.hspace = 0.32

    sum = zeros(length(energy_carriers), nh , length(y), length(s))
    for s_id in s
        for (i, type) in enumerate(energy_carriers)
            i == 1 ? subplot(length(energy_carriers), 1, i, title = string(type)) : subplot(length(energy_carriers), 1, i, sharex = f.axes[1], title = string(type))
            # Demands
            for (k, a) in enumerate(mg.demands)
                if a.carrier isa type
                    sum[i,:,:,s_id] .+= -a.carrier.power[:,y,s_id]
                end
            end
            # Generations
            for (k, a) in enumerate(mg.generations)
                if a.carrier isa type
                    sum[i,:,:,s_id] .+= a.carrier.power[:,y,s_id]
                end
            end
            # Storages
            for (k, a) in enumerate(mg.storages)
                if a.carrier isa type
                    sum[i,:,:,s_id] .+= a.carrier.power[:,y,s_id]
                end
            end
            # Converters
            for (k, a) in enumerate(mg.converters)
                for c in a.carrier
                    if c isa type
                        sum[i,:,:,s_id] .+= c.power[:,y,s_id]
                    end
                end
            end
            for (k, a) in enumerate(mg.grids)
                if a.carrier isa type
                    sum[i,:,:,s_id] .+= a.carrier.power[:,y,s_id]
                end
            end
            Seaborn.plot(hours,  vec(sum[i,:,1:length(y),s_id]), label = string("Net sum"))
            legend()
        end
    end

end

function SoCPlot(mg::Microgrid, y::UnitRange{Int64}, s::UnitRange{Int64}, hours::UnitRange{Int64})
    # State of charge
    f = figure("State-of-charge")

   
    for s_id in s
        for (k, a) in enumerate(mg.storages)
            
            k == 1 ? subplot(length(mg.storages), 1, k) : subplot(length(mg.storages), 1, k, sharex = f.axes[1])
            Seaborn.lineplot(x= hours, y=vec(a.soc[1:end-1, y, s_id]), label = string("Storage : ", typeof(a)))
            legend()

        end
    end
end
 
function SoHPlot(mg::Microgrid, y::UnitRange{Int64}, s::UnitRange{Int64}, hours::UnitRange{Int64})
    #State of health hydrogen
    f = figure("State-of-health")

    assets = vcat(mg.converters,mg.storages, mg.generations)
    #How many of the storages + converters + generators have a SoH field, we keep only those ones
    assets = assets[hasproperty.(assets, :soh)]
   
   
    for (k, a) in enumerate(assets)
        for s_id in s
            k == 1 ? subplot(length(assets), 1, k) : subplot(length(assets), 1, k, sharex = f.axes[1])
            Seaborn.lineplot(x=hours, y=vec(a.soh[1:end-1, y, s_id]), label = string("Storage : ", typeof(a)))
            legend()
        end
    end
end


function plot_operation(mg::Microgrid ; y=2, s=1)
    # Seaborn configuration
    Seaborn.set_theme(context="notebook", style="ticks", palette="muted", font="serif", font_scale=1.5)

    # Parameters
    nh = mg.parameters.nh
    Δh = mg.parameters.Δh
    #Hours range starting from the first year of the interval 
    hours = range((y[1]-1) * nh +1, length = nh * length(y), step = Δh) / Δh

    # we enumerate what type of carrier we have in the converters
    energy_carriers_list = []
    for conv in mg.converters
        for carrier in conv.carrier
            push!(energy_carriers_list, carrier)
        end
    end
    for demand in mg.demands
        push!(energy_carriers_list,  demand.carrier)
    end

    energy_carriers = unique((typeof(a) for a in energy_carriers_list))
    
    # Plots
    # Powers
    powerPlot(energy_carriers, mg, y, s, hours)

    #Power balance to check Electricity and hydrogen perfect balance and heat positivity.
    powerBalancePlot(energy_carriers, mg, y, s, hours)
    
    #State of charge for every storage
    SoCPlot(mg, hours, y, s)

    #State of health for every concerned component
    SoHPlot(mg, hours, y ,s)
   
end


function plot_operation(mg::Microgrid, controller::AbstractController; y=2, s=1)
    # Seaborn configuration
    Seaborn.set_theme(context="notebook", style="ticks", palette="muted", font="serif", font_scale=1.5)

    # Parameters
    nh = mg.parameters.nh
    Δh = mg.parameters.Δh
    hours = range(1, length = nh, step = Δh) / Δh

    # Plots
    # Powers
    f = figure("Powers")
    for (i, type) in enumerate([typeof(Electricity()), typeof(Heat()), typeof(Hydrogen())])
        i == 1 ? subplot(3, 1, i) : subplot(3, 1, i, sharex = f.axes[1])
        # Demands
        for (k, a) in enumerate(mg.demands)
            if a.carrier isa type
                plot(hours, a.carrier.power[:,y,s], label = string("Demand ", k))
            end
        end
        # Generations
        for (k, a) in enumerate(mg.generations)
            if a.carrier isa type
                plot(hours, a.carrier.power[:,y,s], label = string("Generation ", k))
            end
        end
        # Storages
        for (k, a) in enumerate(mg.storages)
            if a.carrier isa type
                plot(hours, controller.decisions.storages[k][:,y,s], label = string("Storage ", k))
            end
        end
        # Converters
        for (k, a) in enumerate(mg.converters)
            for c in a.carrier
                if c isa type
                    plot(hours, controller.decisions.converters[k][:,y,s], label = string("Converter ", k))
                end
            end
        end
        for (k, a) in enumerate(mg.grids)
            if a.carrier isa type
                plot(hours, a.carrier.power[:,y,s], label = string("Grids ", k))
            end
        end
        legend()
    end
end
# Statistics
function plot_metrics(metrics::Metrics; type = "hist")
    # Seaborn configuration
    Seaborn.set_theme(context="notebook", style="ticks", palette="muted", font="serif", font_scale = 1.5)

    if type == "hist"
        figure("Renewable share")
        hist(reshape(metrics.renewable_share[2:end, :], :) * 100)
        ylabel("Counts", size = "large"), yticks(size = "medium")
        xlabel("Renewable share (%)", size = "large"), xticks(size = "medium")
        figure("Cost")
        hist(reshape(metrics.eac.total[:, :], :) / 1000)
        ylabel("Counts", size = "large"), yticks(size = "medium")
        xlabel("Annual cost (k€/y)", size = "large"), xticks(size = "medium")
        if !isa(metrics.lpsp.elec, Nothing)
            figure("LPSP elec")
            hist(reshape(metrics.lpsp.elec[2:end, :], :) * 100)
            ylabel("Counts", size = "large"), yticks(size = "medium")
            xlabel("LPSP (%)", size = "large"), xticks(size = "medium")
        end
        if !isa(metrics.lpsp.heat, Nothing)
            figure("LPSP heat")
            hist(reshape(metrics.lpsp.heat[2:end, :], :) * 100)
            ylabel("Counts", size = "large"), yticks(size = "medium")
            xlabel("LPSP (%)", size = "large"), xticks(size = "medium")
        end
        if !isa(metrics.lpsp.EnergyCarrier, Nothing)
            figure("LPSP EnergyCarrier")
            hist(reshape(metrics.lpsp.EnergyCarrier[2:end, :], :) * 100)
            ylabel("Counts", size = "large"), yticks(size = "medium")
            xlabel("LPSP (%)", size = "large"), xticks(size = "medium")
        end
    elseif type == "violin"
        figure("Renewable share")
        violinplot(reshape(metrics.renewable_share[2:end, :], :) * 100)
        yticks(size = "medium")
        xlabel("Renewable share (%)", size = "large"), xticks(size = "medium")
        figure("Cost")
        violinplot(reshape(metrics.eac.total[:, :], :) / 1000)
        yticks(size = "medium")
        xlabel("Annual cost (k€/y)", size = "large"), xticks(size = "medium")
        if !isa(metrics.lpsp.elec, Nothing)
            figure("LPSP elec")
            violinplot(reshape(metrics.lpsp.elec[2:end, :], :) * 100)
            yticks(size = "medium")
            xlabel("LPSP (%)", size = "large"), xticks(size = "medium")
        end
        if !isa(metrics.lpsp.heat, Nothing)
            figure("LPSP heat")
            violinplot(reshape(metrics.lpsp.heat[2:end, :], :) * 100)
            yticks(size = "medium")
            xlabel("LPSP (%)", size = "large"), xticks(size = "medium")
        end
        if !isa(metrics.lpsp.EnergyCarrier, Nothing)
            figure("LPSP EnergyCarrier")
            violinplot(reshape(metrics.lpsp.EnergyCarrier[2:end, :], :) * 100)
            yticks(size = "medium")
            xlabel("LPSP (%)", size = "large"), xticks(size = "medium")
        end
    else
        println("Only 'hist' or 'violin' type accepted")
    end
end
