#=
    This file includes all the funtions needed to compute the techno-economic
    indicators
 =#

#Power subscription prices interpatated/extrapolated to be on a continous scale.
# Prices comes from https://www.kelwatt.fr/guide/heures-creuses#:~:text=La%20p%C3%A9riode%20des%20heures%20creuses,et%20entre%2020h%20et%208h.
power_sub = [6, 9, 12, 15, 18, 24, 30, 36]
prices = [144.32, 183.63, 221.81, 258.98, 294.25, 360.61, 428.27, 494.92]
interp_linear_extrap_sub_prices = LinearInterpolation(power_sub, prices, extrapolation_bc=Line())

mutable struct COST{T <: Array{Float64}}
    capex::T
    opex::T
    total::T
    salvage::T
end

mutable struct TCO{T <: Array{Float64}}
    capex::T
    opex::T
    salvage::T
    total::T
end

# TCO take account of grid cost, capex, salvage
TCO(mg::Microgrid, designer::AbstractDesigner) = TCO(1:mg.parameters.ns, mg, designer)
function TCO(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)

    capexx = capex(s, mg, designer)

    opex = dropdims(grid_cost(s, mg, designer),dims=1)

    salvage = salvage_value(s, mg)

    return TCO(capexx, opex, salvage, capexx .+ opex .- salvage)
end


# Compute costs
COST(mg::Microgrid, designer::AbstractDesigner) = COST(1:mg.parameters.ns, mg, designer)
# Compute costs for a given scenario s
# COST take account of grid cost, capex, actualization rate and salvage
function COST(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)

    γ = repeat(1. ./ (1. + mg.parameters.τ) .^ range(0, length = mg.parameters.ny, step = mg.parameters.Δy), 1, length(s))

    capexx = γ .* capex(s, mg, designer)

    opex = γ .* dropdims(complex_grid_cost(s, mg, designer),dims=1)

    salvage = γ .*  salvage_value(s, mg)

    return COST(capexx, opex, capexx .+ opex .- salvage, salvage )
end

 mutable struct NPV{T <: Array{Float64}}
      capex::T
      opex::T
      salvage::T
      cf::T
      cumulative_npv::T
      total::T
 end

 # Compute costs
NPV(mg::Microgrid, designer::AbstractDesigner) = NPV(1:mg.parameters.ns, mg, designer)
# Compute costs for a given scenario s
# NPV take account of  grid cost, capex, baseline, actualization rate and salvage
function NPV(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)

    # Discount factor
    γ = repeat(1. ./ (1. + mg.parameters.τ) .^ range(0, length = mg.parameters.ny, step = mg.parameters.Δy), 1, length(s))

    # Discounted capex
    capexx = γ .* capex(s, mg, designer)

    # Discounted opex
    #opex = γ .* dropdims(baseline_cost(s, mg) .-  grid_cost(s, mg, designer), dims=1)
    opex = γ .* dropdims(baseline_cost(s, mg) .-  complex_grid_cost(s, mg, designer), dims=1)

    # Discounted salvage value
    salvage = γ .*  salvage_value(s, mg)

    # Discounted cash flow
    cf = - capexx .+ opex .+ salvage

    # Discounted NPV each year
    cumulative = cumsum(cf, dims = 1)

    # Discounted NPV
    total = sum(cf, dims=1)

    return NPV(capexx, opex, salvage, cf, cumulative, total)
end

# Baseline cost
# The cost of fulfilling all demand with the grid.
baseline_cost(mg::Microgrid) = baseline_cost(1:mg.parameters.ns, mg)
baseline_cost(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid) = baseline_cost(1:mg.parameters.ny, s, mg)
function baseline_cost(y::Union{Int64, UnitRange{Int64}}, s::Union{Int64, UnitRange{Int64}}, mg::Microgrid)
    # TODO : compute baseline cost whatever the microgrid...

    total = 0.
    if !isempty(mg.demands)
        for (k,a) in enumerate(mg.demands)
            if a.carrier isa Electricity
                #
                total = total .+ sum(a.carrier.power[:,y,s] .* mg.grids[1].cost_in[:,y,s] * mg.parameters.Δh, dims = 1)

                #Hypothese l'abonnement était designé parfaitement pour l'utilisation
                # Entrainant 0 coût de depassement
                P_max = maximum(a.carrier.power[:,y,s], dims = 1)
                subscribtion = zeros(1,length(y), length(s))
                for i in y
                    for j in s
                        subscribtion[:,i,j] = interp_linear_extrap_sub_prices(P_max[:,i,j])
                    end
                end
                total = total .+ subscribtion  #abonnement

            elseif a.carrier isa Heat
                total = total .+ sum(a.carrier.power[:,y,s] / mg.converters[isin(mg.converters, Heater)[2]].η_E_H .* mg.grids[1].cost_in[:,y,s] * mg.parameters.Δh, dims = 1)
            end
        end
    end


    return total
end
# Grid cost
grid_cost(mg::Microgrid, designer::AbstractDesigner) = grid_cost(1:mg.parameters.ns, mg , designer)
grid_cost(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner) = grid_cost(1:mg.parameters.ny, s, mg, designer)
function grid_cost(years::Union{Int64, UnitRange{Int64}}, s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)

    return sum(sum(max.(0., a.carrier.power[:,years,s]) .* a.cost_in[:,years,s] .+ min.(0., a.carrier.power[:,years,s]) .* a.cost_out[:,years,s], dims = 1) * mg.parameters.Δh for a in mg.grids) # Energy buying cost

end


complex_grid_cost(mg::Microgrid, designer::AbstractDesigner) = complex_grid_cost(1:mg.parameters.ns, mg , designer)
complex_grid_cost(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner) = complex_grid_cost(1:mg.parameters.ny, s, mg, designer)
function complex_grid_cost(years::Union{Int64, UnitRange{Int64}}, scenarios::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)

    nh = mg.parameters.nh
   
    #Energy buying cost
    net_energy_cost = sum(sum(max.(0., a.carrier.power[:,years,scenarios]) .* a.cost_in[:,years,scenarios] .+ min.(0., a.carrier.power[:,years,scenarios]) .* a.cost_out[:,years,scenarios], dims = 1) * mg.parameters.Δh for a in mg.grids)

    #overcome cost by year
    overcome_cost = zeros(1,length(years),length(scenarios))
    for a in mg.grids
        for s in 1:length(scenarios)
            for y in 1:length(years)
                if a.carrier isa Electricity
                    overcome_cost[1,y,s] = count(nb_overcome->(nb_overcome > 0), a.carrier.power[:,y,s] .- designer.decisions.subscribed_power["Electricity"][y,s]) * a.cost_exceed[y,s] # count the hourly consumption exceeding the subscribed power
                elseif a.carrier isa Heat
                    overcome_cost[1,y,s] = count(nb_overcome->(nb_overcome > 0), a.carrier.power[:,y,s] .- designer.decisions.subscribed_power["Heat"][y,s]) * a.cost_exceed[y,s]  # count the hourly consumption exceeding the subscribed power
                elseif a.carrier isa Hydrogen
                    overcome_cost[1,y,s] = count(nb_overcome->(nb_overcome > 0), a.carrier.power[:,y,s] .- designer.decisions.subscribed_power["Hydrogen"][y,s]) * a.cost_exceed[y,s] # count the hourly consumption exceeding the subscribed power
                end
            end
        end
    end

    net_energy_cost .+= overcome_cost
    return net_energy_cost
end

# CAPEX
capex(mg::Microgrid, designer::AbstractDesigner) = capex(1:mg.parameters.ns, mg, designer)
# CAPEX for a given scenario s
function capex(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)



    capexx = 0.
    # Generations
    for (k, a) in enumerate(mg.generations)
        capexx = capexx .+ designer.decisions.generations[string(typeof(a))][:,s] .* a.cost[:,s]
        capexx[1,s] = capexx[1,s] + designer.generations[string(typeof(a))] * a.cost[1,s]

    end
    # Storages
    for (k, a) in enumerate(mg.storages)
        capexx[1,s] = capexx[1,s] + designer.storages[string(typeof(a))] * a.cost[1,s]
        capexx = capexx .+ designer.decisions.storages[string(typeof(a))][:,s] .* a.cost[:,s]
    end
    # Converters
    for (k, a) in enumerate(mg.converters)
        if a isa FuelCell || a isa Electrolyzer
            key = string(typeof(a))
           
            P_nom_replacement = maximum(a.V_J_ini[1,:] .* a.V_J_ini[2,:]) * designer.decisions.converters[key].surface .* designer.decisions.converters[key].N_cell
            P_nom_ini = maximum(a.V_J_ini[1,:] .* a.V_J_ini[2,:]) * designer.converters[key].surface .* designer.converters[key].N_cell

            #Initial installation
            capexx[1,s] = capexx[1,s] .+ P_nom_ini .* a.cost[1,s]
            #Each replacement
            capexx = capexx .+ P_nom_replacement[:,s] .* a.cost[:,s]
            
        elseif a isa Heater
            capexx[1,s] = capexx[1,s] + designer.converters[string(typeof(a))] * a.cost[1,s]
            capexx = capexx .+ designer.decisions.converters[string(typeof(a))][:,s] .* a.cost[:,s]
        end


    end


    # Subscribtion grid
    for a in mg.grids
        subscribtion = zeros(size(designer.decisions.subscribed_power[string(typeof(a.carrier))][:,s]))
        for i in 1:size(designer.decisions.subscribed_power[string(typeof(a.carrier))][:,s], 1)
            for j in s
                subscribtion[i,j] = interp_linear_extrap_sub_prices(designer.decisions.subscribed_power[string(typeof(a.carrier))][i,j])
            end
        end
        capexx = capexx .+ subscribtion  #abonnement
    end
    return capexx
end

# Salvage value
salvage_value(mg::Microgrid) = salvage_value(1:mg.parameters.ns, mg)
# Salvage value for a given scenario s
function salvage_value(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid)
    # Linear depreciation of components
    nh, ny = mg.parameters.nh, mg.parameters.ny
    salvage = zeros(mg.parameters.ny, length(s))
    salvage[ny,:] .= 0.
    # Generations
    for a in mg.generations
        salvage[ny,:] = salvage[ny,:] .+ (a.SoH_model.lifetime .- ny%a.SoH_model.lifetime) ./ a.SoH_model.lifetime .* a.cost[ny, s] .* a.powerMax[ny,s]
    end
    # Storages
    for a in mg.storages
        if hasproperty(a, :soh)
            salvage[ny,:] = salvage[ny,:] .+ ((a.soh[1,end,s] .- a.SoH_threshold) ./ (1 .-a.SoH_threshold)) .* a.cost[ny, s] .* a.Erated[ny,s] # 100% value at 100% SOH, 0% at EOL
            #salvage[ny,:] = salvage[ny,:] .+ a.soh[1,end,s] .* a.cost[ny, s] .* a.Erated[ny,s]
            #$a.soh[end,end,s]$ remplace ici $(a.lifetime .- ny) ./ a.lifetime$ comme indicateur de la fraction de vie restante
        else
            salvage[ny,:] = salvage[ny,:] .+ (a.SoH_model.lifetime .- ny%a.SoH_model.lifetime) ./ a.SoH_model.lifetime .* a.cost[ny, s] .* a.Erated[ny,s]
        end
    end
    # Converters
    for a in mg.converters
        if hasproperty(a, :soh)
            if a isa AbstractFuelCell || a isa AbstractElectrolyzer
                P_nom = maximum(a.V_J_ini[1,:] .* a.V_J_ini[2,:]) .* a.surface .* a.N_cell
                # Erreur possible car on pourrait avoir soh et pas N_cell et surface
                salvage[ny,:] = salvage[ny,:] .+ ((a.soh[1,end,s] .- a.SoH_threshold) ./ (1 .-a.SoH_threshold)) .* a.cost[ny, s]  .* P_nom
            end
        else
            salvage[ny,:] = salvage[ny,:] .+ (a.SoH_model.lifetime .- ny%a.SoH_model.lifetime) ./ a.SoH_model.lifetime .* a.cost[ny, s]
        end
    end
   
    return salvage
end

mutable struct EAC{T <: Array{Float64}}
     capex::T
     opex::T
     total::T
end

function EAC(y::Int64, s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)
    # Annualised capex
    capex = annualised_capex(1:y, s, mg, designer)
    # opex
    opex = grid_cost(y, s, mg, designer)

    return EAC(capex, opex, capex .+ opex)
end
# Annualised CAPEX
function annualised_capex(y::Union{Int64, UnitRange{Int64}}, s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)
    # Preallocation
    capex = 0.
    # Generations
    for (k, a) in enumerate(mg.generations)
    
        Γ = (mg.parameters.τ * (mg.parameters.τ + 1.) ^ a.SoH_model.lifetime) / ((mg.parameters.τ + 1.) ^ a.SoH_model.lifetime - 1.)            
        capex = capex .+ Γ .* designer.decisions.generations[string(typeof(a))][y,s] .* a.cost[y,s]
        capex[1,s] = capex[1,s] + Γ[1] .* designer.generations[string(typeof(a))] * a.cost[1,s]

    end
    # Storages
    for (k, a) in enumerate(mg.storages)
        if hasproperty(a, :soh)
            id = findfirst(a.soh[:,:,s] .<= a.SoH_threshold) 
            lifetime = id[2] + id[1]/8760
        else
            lifetime = a.SoH_model.lifetime
        end
        Γ = (mg.parameters.τ * (mg.parameters.τ + 1.) ^ lifetime) / ((mg.parameters.τ + 1.) ^  lifetime - 1.)
        capex = capex .+ Γ .* designer.decisions.storages[string(typeof(a))][y,s] .* a.cost[y,s]
        capex[1,s] = capex[1,s] + Γ[1] .* designer.storages[string(typeof(a))] * a.cost[1,s]

    end
    # Converters
    for (k, a) in enumerate(mg.converters)
        if a isa FuelCell ||a isa Electrolyzer
            id = findfirst(a.soh[:,:,s] .<= a.SoH_threshold) 
            lifetime = id[2] + id[1]/8760
        else
            lifetime = a.SoH_model.lifetime
        end

        Γ = (mg.parameters.τ * (mg.parameters.τ + 1.) ^ lifetime) / ((mg.parameters.τ + 1.) ^ lifetime - 1.)

        if a isa FuelCell
            key = "FuelCell"
            
            P_nom_replacement = maximum(a.V_J_ini[1,:] .* a.V_J_ini[2,:]) * designer.decisions.converters[key].surface .* designer.decisions.converters[key].N_cell
            P_nom_ini = maximum(a.V_J_ini[1,:] .* a.V_J_ini[2,:]) * designer.converters[key].surface .* designer.converters[key].N_cell

            capex = capex .+ Γ .* P_nom_replacement[y,s] .* a.cost[y,s]
            capex[1,:] = capex[1,:] .+ Γ[1] * P_nom_ini .* a.cost[1,s]
           
        elseif a isa Electrolyzer
            key = "Electrolyzer"
            
            P_nom_replacement = maximum(a.V_J_ini[1,:] .* a.V_J_ini[2,:]) * designer.decisions.converters[key].surface .* designer.decisions.converters[key].N_cell
            P_nom_ini = maximum(a.V_J_ini[1,:] .* a.V_J_ini[2,:]) * designer.converters[key].surface .* designer.converters[key].N_cell

            capex = capex .+ Γ .* P_nom_replacement[y,s] .* a.cost[y,s]
            capex[1,:] = capex[1,:] .+ Γ[1] * P_nom_ini .* a.cost[1,s]
        elseif a isa Heater
            capex = capex .+ Γ .* designer.decisions.converters["Heater"][y,s] .* a.cost[y,s]
            capex[1,:] = capex[1,:] .+ Γ[1] * designer.converters["Heater"] .* a.cost[1,s]
        end
    end

    # Subscribtion grid
    for a in mg.grids
        if a.carrier isa Electricity
            key = "Electricity"
        elseif a.carrier isa Heat
            key = "Heat"
        elseif a.carrier isa hydrogen
            key = "Hydrogen"
        end

        tmp = zeros(size(designer.decisions.subscribed_power[key][:,s]))
        for i in 1:size(designer.decisions.subscribed_power[key][:,s],1)
            for j in s
                tmp[i,j] = interp_linear_extrap_sub_prices(designer.decisions.subscribed_power[key][i,j])
            end
        end
        capex = capex .+ (tmp/length(y))    #abonnement
    end
    return capex
end

# Share of renewables
renewable_share(mg::Microgrid) = renewable_share(1:mg.parameters.ns, mg)
# Share of renewables for a given scenario s
renewable_share(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid) = renewable_share(1:mg.parameters.ny, s, mg)
# Share of renewables for a given year y of a given scenario s
function renewable_share(y::Union{Int64, UnitRange{Int64}}, s::Union{Int64, UnitRange{Int64}}, mg::Microgrid)
    # TODO to be changed if there is no grid...
    total = 0.
    for dem in mg.demands
        if dem.carrier isa Electricity
            total = total .+ sum(dem.carrier.power[:,y,s], dims = 1)
        elseif dem.carrier isa Heat
            total = total .+ sum(dem.carrier.power[:,y,s], dims = 1) ./ mg.converters[isin(mg.converters, Heater)[2]].η_E_H
        end
    end
    for grid in mg.grids
        if grid.carrier isa Electricity
            return share = dropdims(1. .- sum(max.(0., grid.carrier.power[:,y,s]), dims = 1) ./ total, dims=1)
        else
            println("Renewable share not yet defined!")
            return nothing
        end
    end
end

# LPSP
mutable struct LPSP{T}
    elec::Union{Nothing, T}
    heat::Union{Nothing, T}
    EnergyCarrier::Union{Nothing, T}
end

LPSP(mg::Microgrid) = LPSP(1:mg.parameters.ns, mg)
# LPSP for a given scenario s
LPSP(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid) = LPSP(1:mg.parameters.ny, s, mg)
# LPSP for a given scenario s and year y
function LPSP(y::Union{Int64, UnitRange{Int64}}, s::Union{Int64, UnitRange{Int64}}, mg::Microgrid)
    # Initialization
    elec, heat, EnergyCarrier = nothing, nothing, nothing
    # Computation
    for a in mg.demands
        if a.carrier isa Electricity
            elec = sum(max.(0., power_balance(1:mg.parameters.nh, y, s, mg, Electricity)), dims=1) ./ sum(a.carrier.power[:, y, s], dims = 1)
            for aa in mg.grids
                if aa.carrier isa Electricity
                    elec = elec .- sum(max.(0., aa.carrier.power[:, y, s]), dims=1) ./ sum(a.carrier.power[:, y, s], dims = 1)
                end
            end
            elec = dropdims(elec,dims=1)
        elseif a.carrier isa Heat
            heat = sum(max.(0., power_balance(1:mg.parameters.nh, y, s, mg, Heat)), dims=1) ./ sum(a.carrier.power[:, y, s], dims = 1)
            for aa in mg.grids
                if aa.carrier isa Heat
                    heat = heat .- sum(max.(0., aa.carrier.power[:, y, s]), dims=1) ./ sum(a.carrier.power[:, y, s], dims = 1)
                end
            end
            heat = dropdims(heat,dims=1)
        elseif a.carrier isa Hydrogen
            EnergyCarrier = sum(max.(0., power_balance(1:mg.parameters.nh, y, s, mg, Hydrogen)), dims=1) ./ sum(a.carrier.power[:, y, s], dims = 1)
            for aa in mg.grids
                if aa.carrier isa Electricity
                    hydrogen = hydrogen .- sum(max.(0., aa.carrier.power[:, y, s]), dims=1) ./ sum(a.carrier.power[:, y, s], dims = 1)
                end
            end
            hydrogen = dropdims(hydrogen,dims=1)
        end
    end
    return LPSP(elec, heat, EnergyCarrier)
end

mutable struct Metrics{T}
    baseline::T
    npv::NPV{T}
    eac::EAC{T}
    renewable_share::T
    lpsp::LPSP{T}
    cost::COST{T}
    tco::TCO{T}
end

# Compute indicators
Metrics(mg::Microgrid, designer::AbstractDesigner) = Metrics(1:mg.parameters.ns, mg, designer)
# Compute indicators for a given scenario s
function Metrics(s::Union{Int64, UnitRange{Int64}}, mg::Microgrid, designer::AbstractDesigner)
    # Baseline cost
    baseline = dropdims(baseline_cost(mg), dims = 1)
    # NPV
    npv = NPV(s, mg, designer)
    # EAC
    eac = EAC(mg.parameters.ny, s, mg, designer)
    # Share of renewables
    share = renewable_share(s, mg)
    # LPSP
    lpsp = LPSP(s, mg)
    
    #
    cost = COST(s, mg, designer)

    tco = TCO(s, mg, designer)
    

    return Metrics(baseline, npv, eac, share, lpsp, cost, tco)
end


