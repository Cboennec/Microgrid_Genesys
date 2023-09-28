#Lexique des variables : 
# r_g : Vecteur des tailles des générateurs (dim = 1, nombre de générateur)
# r_sto : Vecteur des tailles des Stockers (dim = 1, nombre de stocker)
# r_c : Vecteur des tailles des convertisseurs (dim = 1, nombre de convetisseur)
#
# p_d[1:nh, 1:ns, 1:na] : Matrice des decisions (puissance) de demande (dim = 3, Nombre d'heure, nombre de scénarios, nombre de demande)
# p_g[1:nh, 1:ns, 1:na] : Matrice des decisions (puissance) de générateur (dim = 3, Nombre d'heure, nombre de scénarios, nombre de générateur)
#
# p_ch[1:nh, 1:ns, 1:na]   >= 0. : Matrice des decisions (puissance) de charge des stockeurs (dim = 3, Nombre d'heure, nombre de scénarios, nombre de stockeur)
# p_dch[1:nh, 1:ns, 1:na]  >= 0. : Matrice des decisions (puissance) de décharge des stockeurs (dim = 3, Nombre d'heure, nombre de scénarios, nombre de stockeur)
# soc[1:nh+1, 1:ns, 1:na] : Matrice des niveau de soc des stockeurs (dim = 3, Nombre d'heure + 1, nombre de scénarios, nombre de stockeur)
# 
# p_c[1:nh, 1:ns, 1:na] : Matrice des decisions (puissance) des convertisseurs (dim = 3, Nombre d'heure, nombre de scénarios, nombre de convertisseur)
#
# p_in[1:nh, 1:ns, 1:na]   >= 0. : Matrice des decisions (puissance) vers les grids (dim = 3, Nombre d'heure, nombre de scénarios, nombre de grid)
# p_out[1:nh, 1:ns, 1:na]  >= 0. : Matrice des decisions (puissance) depuis les grids (dim = 3, Nombre d'heure, nombre de scénarios, nombre de grid)



# Risk measures
abstract type AbstractRiskMeasure end
# Expectation
struct Expectation <: AbstractRiskMeasure end
# Worst case
struct WorstCase <: AbstractRiskMeasure end
# Conditional value at risk
struct CVaR <: AbstractRiskMeasure
    β::Float64

    function CVaR(β::Float64)
        @assert 0. <= β <= 1. "β must be in [0,1]"
        return new(β)
    end
end

conditional_value_at_risk(support::Array{Float64,1}, probabilities::Array{Float64,1}, risk::WorstCase) = conditional_value_at_risk(support, probabilities, 0.)
conditional_value_at_risk(support::Array{Float64,1}, probabilities::Array{Float64,1}, risk::Expectation) = conditional_value_at_risk(support, probabilities, 1.)
conditional_value_at_risk(support::Array{Float64,1}, probabilities::Array{Float64,1}, risk::CVaR) = conditional_value_at_risk(support, probabilities, 1. - risk.β)

# From https://github.com/jaantollander/ConditionalValueAtRisk
function conditional_value_at_risk(support::Array{Float64,1}, probabilities::Array{Float64,1}, α::Float64)
    # Value at risk
    var = value_at_risk(support, probabilities, α)
    # Tail values
    if α == 0.
        return var
    else
        tail = support .< var
        return (sum(probabilities[tail] .* support[tail]) - (sum(probabilities[tail]) - α) * var) / α
    end
end

function value_at_risk(support::Array{Float64,1}, probabilities::Array{Float64,1}, α::Float64)

    i = findfirst(cumsum(probabilities[sortperm(support)]) .>= α)
    if i == nothing
        return sort(support)[end]
    else
        return sort(support)[i]
    end
end

# MILP functions
# Decisions
#Investment on the energy generation units (define the size)
function add_investment_decisions!(m::Model, generations::Vector{AbstractGeneration})
    if !isempty(generations)
        na = length(generations)
        @variable(m, r_g[1:na])
    end
end
#Investment on the energy storage units (define the size)
function add_investment_decisions!(m::Model, storages::Vector{AbstractStorage})
    if !isempty(storages)
        na = length(storages)
        @variable(m, r_sto[1:na])
    end
end
#Investment on the energy converter units
function add_investment_decisions!(m::Model, converters::Vector{AbstractConverter})
    if !isempty(converters)
        na = length(converters)
        @variable(m, r_c[1:na])
    end
end


function fix_investment_decisions!(m::Model, generations::Vector{Float64}, storages::Vector{Float64}, converters::Vector{Float64}, mg::Microgrid)
    # Generation
    if !isempty(mg.generations)
        fix.(m[:r_g], generations)
    end
    # Storages
    if !isempty(mg.storages)
        fix.(m[:r_sto], storages)
    end
    # Converters
    if !isempty(mg.converters)
        fix.(m[:r_c], converters)
    end
end

function fix_investment_decisions!(m::Model, generations::Vector{Float64}, storages::Vector{Float64}, converters::Vector{Float64})
    # Generation
    if !isempty(generations)
        fix.(m[:r_g], generations)
    end
    # Storages
    if !isempty(storages)
        fix.(m[:r_sto], storages)
    end
    # Converters
    if !isempty(converters)
        fix.(m[:r_c], converters)
    end
end


function add_operation_decisions!(m::Model, demands::Vector{AbstractDemand}, nh::Int64, ns::Int64)
    if !isempty(demands)
        na = length(demands)
        @variables(m, begin
        p_d[1:nh, 1:ns, 1:na]
        end)
    end
end
function add_operation_decisions!(m::Model, generations::Vector{AbstractGeneration}, nh::Int64, ns::Int64)
    if !isempty(generations)
        na = length(generations)
        @variables(m, begin
        p_g[1:nh, 1:ns, 1:na]
        end)
    end
end
function add_operation_decisions!(m::Model, storages::Vector{AbstractStorage}, nh::Int64, ns::Int64)
    if !isempty(storages)
        na = length(storages)
        @variables(m, begin
        p_ch[1:nh, 1:ns, 1:na]   >= 0.
        p_dch[1:nh, 1:ns, 1:na]  >= 0.
        soc[1:nh+1, 1:ns, 1:na]
        end)
    end
end

function add_operation_decisions!(m::Model, converters::Vector{AbstractConverter}, nh::Int64, ns::Int64)
    if !isempty(converters)
        na = length(converters)
        @variable(m, p_c[1:nh, 1:ns, 1:na] >= 0.)
    end
end
function add_operation_decisions!(m::Model, grids::Vector{AbstractGrid}, nh::Int64, ns::Int64)
    if !isempty(grids)
        na = length(grids)
        @variables(m, begin
        p_in[1:nh, 1:ns, 1:na]   >= 0.
        p_out[1:nh, 1:ns, 1:na]  >= 0.
        end)
    end
end

function add_SoC_base!(m::Model, storages::Vector{AbstractStorage}, ns::Int64)
    if !isempty(storages)
        na = length(storages)

        @variables(m, begin
            SoC_base[1:365, 1:ns, 1:na]   >= 0.
        end)
    end
end


# Investment bounds
function add_investment_constraints!(m::Model, generations::Vector{AbstractGeneration})
    if !isempty(generations)
        na = length(generations)
        @constraints(m, begin
        [a in 1:na], m[:r_g][a] >= generations[a].bounds.lb
        [a in 1:na], m[:r_g][a] <= generations[a].bounds.ub
        end)
    end
end
function add_investment_constraints!(m::Model, storages::Vector{AbstractStorage})
    if !isempty(storages)
        na = length(storages)
        @constraints(m, begin
        [a in 1:na], m[:r_sto][a] >= storages[a].bounds.lb
        [a in 1:na], m[:r_sto][a] <= storages[a].bounds.ub
        end)
    end
end
function add_investment_constraints!(m::Model, converters::Vector{AbstractConverter})
    if !isempty(converters)
        na = length(converters)
        @constraints(m, begin
        [a in 1:na], m[:r_c][a] >= converters[a].bounds.lb
        [a in 1:na], m[:r_c][a] <= converters[a].bounds.ub
        end)
    end
end


# Technical constraint
function add_technical_constraints!(m::Model, storages::Vector{AbstractStorage}, Δh::Int64, nh::Int64, ns::Int64, representative::Bool)
    if !isempty(storages)
        na = length(storages)
        @constraints(m, begin
        # Power bounds
        [h in 1:nh, s in 1:ns, a in 1:na], m[:p_dch][h,s,a] <= storages[a].α_p_dch * m[:r_sto][a]
        [h in 1:nh, s in 1:ns, a in 1:na], m[:p_ch][h,s,a]  <= storages[a].α_p_ch * m[:r_sto][a]
        # SoC bounds
        [h in 1:nh+1, s in 1:ns, a in 1:na], m[:soc][h,s,a] <= storages[a].α_soc_max * m[:r_sto][a]
        [h in 1:nh+1, s in 1:ns, a in 1:na], m[:soc][h,s,a] >= storages[a].α_soc_min * m[:r_sto][a]
        # State dynamics
        # Initial and final states
        soc_ini[s in 1:ns, a in 1:na], m[:soc][1,s,a] == storages[a].soc_ini * m[:r_sto][a]
        end)

        if !representative
            @constraints(m, begin
                [h in 1:nh, s in 1:ns, a in 1:na], m[:soc][h+1,s,a] == m[:soc][h,s,a] * (1. - storages[a].η_self * Δh) - (m[:p_dch][h,s,a] / storages[a].η_dch - m[:p_ch][h,s,a] * storages[a].η_ch) * Δh
            end)
        else
            i=0
            for sto in storages
                i = i+1
                if typeof(sto) <: AbstractFuelCell
                    @constraints(m, begin
                        #[h in 1:nh, s in 1:ns], m[:soc][h+1,s,i] == m[:soc][h,s,i] * (1. - sto.η_self * Δh) - ( factor[h] * (m[:p_dch][h,s,i] / sto.η_dch - m[:p_ch][h,s,i] * sto.η_ch) * Δh )
                        [h in 1:nh, s in 1:ns], m[:soc][h+1,s,i] == m[:soc][h,s,i] * (1. - sto.η_self * Δh) - ((m[:p_dch][h,s,i] / sto.η_dch - m[:p_ch][h,s,i] * sto.η_ch) * Δh )
                    end)
                else
                    @constraints(m, begin
                        [h in 1:nh, s in 1:ns], m[:soc][h+1,s,i] == m[:soc][h,s,i] * (1. - sto.η_self * Δh) - (m[:p_dch][h,s,i] / sto.η_dch - m[:p_ch][h,s,i] * sto.η_ch) * Δh
                    end)
                end
            end
        end
    end
end

function add_SoH_lim_variables_constraint!(m, k, nh, ns, E_ex_tot, lim)
    #Si le stocker est une batterie liion 
    @variables(m, begin
    ΔSoH[1:ns] >= 0.
    end)
    
    @constraints(m, begin   
    #la somme de la degradation calendaire et de l'ensemble des dégradation de cyclage doit être inférieure à lim
        [s in 1:ns], m[:ΔSoH][s] == sum( (m[:p_dch][h,s,k] + m[:p_ch][h,s,k]) / E_ex_tot for h in 1:nh) + (1 - exp(- 4.14e-10 * 3600))*nh
        [s in 1:ns], m[:ΔSoH][s] <= lim
    end)
end

function add_Continuity_SoC_constraints(m::Model, storages::Vector{AbstractStorage}, nh::Int64, ns::Int64, sequence::Vector{Int64})
    ndays = Int(nh/24)
    
    if !isempty(storages)
        na = length(storages)
    end


    @variables(m, begin
    Extrem_high[1:ndays, 1:ns, 1:na] 
    Extrem_low[1:ndays, 1:ns, 1:na] 
    relativ_diff[1:ndays, 1:ns, 1:na]
    end)

    @constraints(m, begin
    # Constraint the Base soc to be a sequence of sum of relative difference
    [d in 1:ndays, s in 1:ns, a in 1:na], m[:relativ_diff][d,s,a] == (m[:soc][d * 24, s, a] - m[:soc][d * 24 - 23, s, a])
    [d in 2:365, s in 1:ns, a in 1:na], m[:SoC_base][d,s,a] == m[:SoC_base][d-1,s,a] + m[:relativ_diff][sequence[d],s,a]
    
    #Constraint extrem values for each days to be greater than the max and min soc diff
    [d in 1:365, h in 1:24, s in 1:ns, a in 1:na], m[:Extrem_high][sequence[d],s,a] >= m[:soc][(sequence[d]-1)*24+h, s, a] - m[:SoC_base][d, s, a] 
    [d in 1:365, h in 1:24, s in 1:ns, a in 1:na], m[:Extrem_low][sequence[d],s,a] <= m[:soc][(sequence[d]-1)*24+h, s, a] - m[:SoC_base][d, s, a] 

    #Constrant Extrem value to be inside to soc  bounds
    [d in 1:365, s in 1:ns, a in 1:na], m[:SoC_base][d,s,a] + m[:Extrem_high][sequence[d],s,a] <= storages[a].α_soc_max * m[:r_sto][a]
    [d in 1:365, s in 1:ns, a in 1:na], m[:SoC_base][d,s,a] + m[:Extrem_low][sequence[d],s,a] <= storages[a].α_soc_min * m[:r_sto][a]

    end)
end


function add_technical_constraints!(m::Model, converters::Vector{AbstractConverter}, nh::Int64, ns::Int64)
    if !isempty(converters)
        na = length(converters)
        @constraints(m, begin
        # Power bounds
        [h in 1:nh, s in 1:ns, a in 1:na], m[:p_c][h,s,a]  <= m[:r_c][a]
        end)
    end
end

function add_technical_constraints!(m::Model, grids::Vector{AbstractGrid}, nh::Int64, ns::Int64)
    if !isempty(grids)
        na = length(grids)
        @constraints(m, begin
        # Power bounds
        [h in 1:nh, s in 1:ns, a in 1:na], m[:p_in][h,s,a]  <= grids[a].powerMax
        [h in 1:nh, s in 1:ns, a in 1:na], m[:p_out][h,s,a] <= grids[a].powerMax
        end)
    end
end




# Periodicity constraint
function add_periodicity_constraints!(m::Model, storages::Vector{AbstractStorage}, ns::Int64)
    # Storages
    if !isempty(storages)
        na = length(storages)
        @constraints(m, begin
        # Final states
        [s in 1:ns, a in 1:na], m[:soc][end,s,a]  >= m[:soc][1,s,a]
        end)
    end
end

# Periodicity constraint
function add_periodicity_constraints_mini!(m::Model, storages::Vector{AbstractStorage}, ns::Int64)
    # Storages
    if !isempty(storages)
        na = length(storages)
        @constraints(m, begin
        # Final states
        [s in 1:ns, a in 1:na], m[:soc][end,s,a]  >= m[:soc][1,s,a]
        [s in 1:ns, a in 1:na], m[:SoC_base][end,s,a] + m[:relativ_diff][end,s,a] >= m[:SoC_base][1,s,a]
        end)
    end
end





# Power balance
function add_power_balance!(m::Model, mg::Microgrid, ω::AbstractScenarios, type::DataType, nh::Int64, ns::Int64; ispnet::Bool=false)
    # !!! All the decision variables are defined positive !!!
    balance = AffExpr.(zeros(nh,ns))
    # Demands and generation
    if !ispnet
        for (k,a) in enumerate(mg.demands)
            if a.carrier isa type
                add_to_expression!.(balance, ω.demands[k].power[:,1,:])
            end
        end
        # Generation
        for (k,a) in enumerate(mg.generations)
            if a.carrier isa type
                add_to_expression!.(balance, .- m[:r_g][k] .* ω.generations[k].power[:,1,:])
            end
        end
    else
        for (k,a) in enumerate(mg.demands)
            if a.carrier isa type
                add_to_expression!.(balance, m[:p_d][:,:,k])
            end
        end
        # Generation
        for (k,a) in enumerate(mg.generations)
            if a.carrier isa type
                add_to_expression!.(balance, .- m[:p_g][:,:,k])
            end
        end
    end
    # Storages
    for (k,a) in enumerate(mg.storages)
        if a.carrier isa type
            add_to_expression!.(balance, m[:p_ch][:,:,k] .- m[:p_dch][:,:,k])
        end
    end
    # Converters
    for (k,a) in enumerate(mg.converters)
        if type == Electricity
            if a isa Heater
                add_to_expression!.(balance, m[:p_c][:,:,k])
            elseif a isa Electrolyzer
                add_to_expression!.(balance, m[:p_c][:,:,k])
            elseif a isa AbstractFuelCell
                add_to_expression!.(balance, .- m[:p_c][:,:,k])
            end
        elseif type == Heat
            if a isa Heater
                add_to_expression!.(balance, .- m[:p_c][:,:,k] * a.η_E_H)
            elseif a isa Electrolyzer
                add_to_expression!.(balance, .- m[:p_c][:,:,k] * a.η_E_H)
            elseif a isa AbstractFuelCell
                add_to_expression!.(balance, .- m[:p_c][:,:,k] * a.η_H2_H / a.η_H2_E)
            end
        elseif type == Hydrogen
            if a isa Electrolyzer
                add_to_expression!.(balance, .- m[:p_c][:,:,k] * a.η_E_H2)
            elseif a isa AbstractFuelCell
                add_to_expression!.(balance, m[:p_c][:,:,k] / a.η_H2_E)
            end
        end
    end
    # Grids
    for (k,a) in enumerate(mg.grids)
        if a.carrier isa type
            add_to_expression!.(balance, .- m[:p_in][:,:,k] + m[:p_out][:,:,k])
        end
    end
    # Energy balance constraint
    if type == Electricity
        @constraint(m, electricity, balance .<= 0.)
    elseif type == Heat
        @constraint(m, heat, balance .<= 0.)
    elseif type == Hydrogen
        @constraint(m, hydrogen, balance .== 0.)
    end
end


# Renewable share
function add_renewable_share!(m::Model, mg::Microgrid, ω::AbstractScenarios, probabilities::Vector{Float64}, risk::AbstractRiskMeasure, nh::Int64, ns::Int64)
    total = zeros(ns)
    for (k,a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            total .= total .+ sum(ω.demands[k].power[h,1,:] for h in 1:nh)
        elseif a.carrier isa Heat
            total .= total .+ sum(ω.demands[k].power[h,1,:] for h in 1:nh) ./ mg.converters[isin(mg.converters, Heater)[2]].η_E_H
        end
    end
    for (k,a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            @expression(m, share[s in 1:ns], sum(m[:p_in][h,s,k] for h in 1:nh) - (1. - mg.parameters.renewable_share) * total[s])
        end
    end
    # Constraint according to CVaR
    @variables(m, begin
    ζ_s
    α_s[1:ns] >= 0.
    end)
    @constraints(m, begin
    [s in 1:ns], α_s[s] >= m[:share][s] - ζ_s
    ζ_s + 1 / (1 - beta(risk)) * sum(probabilities[s] * α_s[s] for s in 1:ns) <= 0.
    end)
end



# Renewable share
function add_renewable_share_robust!(m::Model, mg::Microgrid, ω::AbstractScenarios, probabilities::Vector{Float64}, risk::AbstractRiskMeasure, nh::Int64, ny::Int64, ns::Int64)
    total_conso_elec = zeros(ns)
    for (k,a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            total_conso_elec .= total_conso_elec .+ sum(ω.demands[k].power[h,y,:] for h in 1:nh for y in 1:ny)
        elseif a.carrier isa Heat
            total_conso_elec .= total_conso_elec .+ sum(ω.demands[k].power[h,y,:] for h in 1:nh for y in 1:ny) ./ mg.converters[isin(mg.converters, Heater)[2]].η_E_H
        end
    end

    share = zeros(ns)
    for (k,a) in enumerate(mg.grids)
        if a.carrier isa Electricity
             share = 1. .- sum(m[:p_in][h,y,:,k] for h in 1:nh for y in 1:ny) ./ total_conso_elec
        end
    end

    @constraints(m, begin
    # Power bounds
    [s in 1:ns], share[s] >= mg.parameters.renewable_share
    end)
end



# Renewable share
function add_renewable_share_robust!(m::Model, mg::Microgrid, ω::AbstractScenarios, nh::Int64, ns::Int64)
    total_conso_elec = zeros(ns)
    for (k,a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            total_conso_elec .= total_conso_elec .+ sum(ω.demands[k].power[h,1,:] for h in 1:nh)
        elseif a.carrier isa Heat
            total_conso_elec .= total_conso_elec .+ sum(ω.demands[k].power[h,1,:] for h in 1:nh) ./ mg.converters[isin(mg.converters, Heater)[2]].η_E_H
        end
    end

    #share = zeros(ns)
    for (k,a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            @expression(m, sum_grid[s in 1:ns], sum(m[:p_in][h,s,k] for h in 1:nh))
             #share = 1. .- sum(m[:p_in][h,:,k] for h in 1:nh) ./ total_conso_elec
        end
    end

    @constraints(m, begin
    # Power bounds
    [s in 1:ns], m[:sum_grid][s] <= (1. - mg.parameters.renewable_share) * total_conso_elec[s]
    end)
end


# Objective
function add_design_objective!(m::Model, mg::Microgrid, ω::AbstractScenarios, probabilities::Vector{Float64}, risk::AbstractRiskMeasure, nh::Int64, ns::Int64)
    # CAPEX
    capex = compute_capex(m, mg, ω)
    # OPEX
    opex = compute_opex(m, mg, ω, nh, ns)
    # Objective according to the CVaR
    @variables(m, begin
    ζ_o
    α_o[1:ns] >= 0.
    end)
    @constraint(m, [s in 1:ns], α_o[s] >= capex + opex[s] - ζ_o)
    @objective(m, Min, ζ_o + 1 / (1 - beta(risk)) * sum(probabilities[s] * α_o[s] for s in 1:ns))
end


# Capex
function compute_capex(m::Model, mg::Microgrid, ω::AbstractScenarios)
    cost = AffExpr(0.)
    # Generations
    for (k,a) in enumerate(mg.generations)
        add_to_expression!(cost, Γ(mg.parameters.τ, a.lifetime) * ω.generations[k].cost[1] * m[:r_g][k])
    end
    # Storages
    for (k,a) in enumerate(mg.storages)
        add_to_expression!(cost, Γ(mg.parameters.τ, a.lifetime) * ω.storages[k].cost[1] * m[:r_sto][k])
    end
    # Converters
    for (k,a) in enumerate(mg.converters)
        add_to_expression!(cost, Γ(mg.parameters.τ, a.lifetime) * ω.converters[k].cost[1] * m[:r_c][k])
    end
    return cost
end

# Grids
function compute_opex(m::Model, mg::Microgrid, ω::AbstractScenarios, nh::Int64, ns::Int64)
    cost = AffExpr.(zeros(ns))
    for (k,a) in enumerate(mg.grids)
        add_to_expression!.(cost, sum((m[:p_in][h,:,k] .* ω.grids[k].cost_in[h,1,:] .- m[:p_out][h,:,k] .* ω.grids[k].cost_out[h,1,:]) .* mg.parameters.Δh  for h in 1:nh))
    end
    return cost
end



function compute_salvage(m::Model, mg::Microgrid, ω::AbstractScenarios, ny::Int64, ns::Int64)
    salvage =  AffExpr.(zeros(ns))
    K = ω.storages[1].cost[ny] * m[:E_state][end]
    add_to_expression!.(salvage, K * m[:soh][end,end,:])

    return salvage
end





############### Special multi year #################

#Operation variables
function add_operation_decisions_my!(m::Model, storages::Vector{AbstractStorage}, generations::Vector{AbstractGeneration},  converters::Vector{AbstractConverter}, grids::Vector{AbstractGrid}, nh::Int64, ny::Int64, ns::Int64)
 
    if !isempty(generations)
        na = length(generations)
        @variables(m, begin
        p_g[1:nh, 1:ny, 1:ns, 1:na]
        end)
    end

    if !isempty(storages)
        na = length(storages)
        @variables(m, begin
        p_ch[1:nh, 1:ny, 1:ns, 1:na]   >= 0.
        p_dch[1:nh, 1:ny, 1:ns, 1:na]  >= 0.
        soc[1:nh+1, 1:ny+1, 1:ns, 1:na]
        end)
    end


    if !isempty(converters)
        na = length(converters)
        @variable(m, p_c[1:nh, 1:ny, 1:ns, 1:na] >= 0.)
    end

    if !isempty(grids)
        na = length(grids)
        @variables(m, begin
        p_in[1:nh, 1:ny, 1:ns, 1:na]   >= 0.
        p_out[1:nh, 1:ny, 1:ns, 1:na]  >= 0.
        end)
    end

    #Replacement battery
    #Battery capacity
    #Battery efficiency
    #Battery SoH
    @variables(m, begin
    δbat[1:ny, 1:ns], Bin 
    E_bat[1:(ny+1), 1:ns] >= 0.
    η_bat[1:(ny+1), 1:ns] >= 0.
    soh[1:nh+1, 1:ny+1, 1:ns] >= 0.
    end)
 

end
   


# Technical constraints
function add_technical_constraints_my!(m::Model, storages::Vector{AbstractStorage}, Δh::Int64, nh::Int64, ny::Int64, ns::Int64)
    not_a_liion = [!(storages[i] isa AbstractLiion) for i in 1:length(storages)]
    storages_without_liion = storages[not_a_liion]
    storages_liion = storages[.!not_a_liion]

    id_liion = (1:length(storages))[.!not_a_liion][1]
    id_not_liion = (1:length(storages))[not_a_liion]

    if !isempty(storages_without_liion)
        na = length(storages_without_liion)
        @constraints(m, begin
        # Power bounds
        [h in 1:nh, y in 1:ny, s in 1:ns, a in id_not_liion], m[:p_dch][h,y,s,a] <= storages[a].α_p_dch * m[:r_sto][a]
        [h in 1:nh, y in 1:ny, s in 1:ns, a in id_not_liion], m[:p_ch][h,y,s,a]  <= storages[a].α_p_ch * m[:r_sto][a]
        # SoC bounds
        [h in 1:nh+1, y in 1:ny, s in 1:ns, a in id_not_liion], m[:soc][h,y,s,a] <= storages[a].α_soc_max * m[:r_sto][a]
        [h in 1:nh+1, y in 1:ny, s in 1:ns, a in id_not_liion], m[:soc][h,y,s,a] >= storages[a].α_soc_min * m[:r_sto][a]
        # State dynamics
        [h in 1:nh, y in 1:ny, s in 1:ns, a in id_not_liion], m[:soc][h+1,y,s,a] == m[:soc][h,y,s,a] * (1. - storages[a].η_self * Δh) - (m[:p_dch][h,y,s,a] / storages[a].η_dch - m[:p_ch][h,y,s,a] * storages[a].η_ch) * Δh
        # Initial and final states
        soc_ini[s in 1:ns, a in id_not_liion], m[:soc][1,1,s,a] == storages[a].soc_ini * m[:r_sto][a]
        end)
    end


    #Liion Part
    if !isempty(storages_liion)
        na = length(storages_liion)
        @constraints(m, begin
        # Power bounds
        [h in 1:nh, y in 1:ny, s in 1:ns], m[:p_dch][h,y,s,id_liion] <= storages[id_liion].α_p_dch * m[:E_bat][y,s]
        [h in 1:nh, y in 1:ny, s in 1:ns], m[:p_ch][h,y,s,id_liion]  <= storages[id_liion].α_p_ch * m[:E_bat][y,s]
        # SoC bounds
        [h in 1:nh+1, y in 1:ny, s in 1:ns], m[:soc][h,y,s,id_liion] <= storages[id_liion].α_soc_max * m[:E_bat][y,s]
        [h in 1:nh+1, y in 1:ny, s in 1:ns], m[:soc][h,y,s,id_liion] >= storages[id_liion].α_soc_min * m[:E_bat][y,s]
   
        # Initial states
        soc_ini_liion[s in 1:ns], m[:soc][1,1,s,id_liion] == storages[id_liion].soc_ini * m[:r_sto][id_liion]
        soh_ini[s in 1:ns], m[:soh][1,1,s] == storages[1].soh_ini * m[:r_sto][id_liion]
        
        
        #Replacement
        replacement[y in 1:ny, s in 1:ns], m[:δbat][y,s] >= (storages[id_liion].SoH_threshold * m[:r_sto][id_liion]) - m[:soh][nh,y,s]
        [s in 1:ns], m[:δbat][1,s] == 1
        
        end)
    end

    # State dynamics
    if !isempty(storages_liion)
        @NLconstraint(m, [h in 1:nh, y in 1:ny, s in 1:ns],
         m[:soc][h+1,y,s,1] == 
         m[:soc][h,y,s,1] * (1. - storages[id_liion].η_self * Δh) - (m[:p_dch][h,y,s,1] / m[:η_bat][y,s] - m[:p_ch][h,y,s,1] * m[:η_bat][y,s]) * Δh
         )
        @NLconstraint(m, [h in 1:nh, y in 1:ny, s in 1:ns], 
        m[:soh][h+1,y,s,1] ==      
        m[:soh][h,y,s,1]  - 
        (
            (m[:p_dch][h,y,s,1] + m[:p_ch][h,y,s,1])/(m[:r_sto][id_liion] * 2 * storages[id_liion].nCycle * (storages[id_liion].α_soc_max - storages[id_liion].α_soc_min))
            +  (1 - exp(- 4.14e-10 * 3600 * Δh)))
            * (Δh * m[:r_sto][id_liion])
        ) 
    end

end


function add_coupling_constraints_my!(m::Model, storages::Vector{AbstractStorage}, nh::Int64, ny::Int64, ns::Int64)
    not_a_liion = [!(storages[i] isa AbstractLiion) for i in 1:length(storages)]
    storages_liion = storages[.!not_a_liion][1]

    a = (1:length(storages))[.!not_a_liion][1]

    @constraints(m, begin

    [s in 1:ns], m[:E_bat][1,s] == m[:r_sto][a]
    #Capacity replacement and coupling M=m[:r_sto][a]
    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - m[:r_sto][a] >= -m[:r_sto][a] * (1-m[:δbat][y,s]) 
    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - m[:r_sto][a] <= m[:r_sto][a] * (1-m[:δbat][y,s]) 

    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - m[:soh][nh,y,s] >= -m[:r_sto][a] * m[:δbat][y,s] 
    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - m[:soh][nh,y,s] <= m[:r_sto][a] * m[:δbat][y,s] 

    
    [s in 1:ns],  m[:η_bat][1,s] == storages_liion.η_ch
    #Efficiency replacement and coupling M=1
    [y in 1:ny, s in 1:ns],  m[:η_bat][y+1,s] - storages_liion.η_ch >= -(1-m[:δbat][y,s]) 
    [y in 1:ny, s in 1:ns],  m[:η_bat][y+1,s] - storages_liion.η_ch <=  (1-m[:δbat][y,s]) 


    #In case of replacement, new soc, soh
    #SoC M=1
    [y in 1:ny, s in 1:ns],  m[:soc][1,y+1,s,a] - (storages_liion.soc_ini * m[:r_sto][a]) >= -(1-m[:δbat][y,s]) 
    [y in 1:ny, s in 1:ns],  m[:soc][1,y+1,s,a] - (storages_liion.soc_ini * m[:r_sto][a]) <= (1-m[:δbat][y,s]) 

    [y in 1:ny, s in 1:ns],  m[:soc][1,y+1,s,a] - m[:soc][nh,y,s,a] >= -m[:δbat][y,s] 
    [y in 1:ny, s in 1:ns],  m[:soc][1,y+1,s,a] - m[:soc][nh,y,s,a] <= m[:δbat][y,s] 

    #SoH M=1
    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - (storages_liion.soh_ini * m[:r_sto][a]) >= -(1-m[:δbat][y,s]) 
    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - (storages_liion.soh_ini * m[:r_sto][a]) <= (1-m[:δbat][y,s]) 

    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - m[:soh][nh,y,s] >= -m[:δbat][y,s] 
    [y in 1:ny, s in 1:ns],  m[:soh][1,y+1,s] - m[:soh][nh,y,s] <= m[:δbat][y,s] 
    end)


    @NLconstraint(m, [y in 1:ny, s in 1:ns],  m[:η_bat][y+1,s] - (storages_liion.η_ch - .25*(1-m[:soh][nh,y,s]/m[:r_sto][a])) >= -m[:δbat][y,s])
    @NLconstraint(m, [y in 1:ny, s in 1:ns],  m[:η_bat][y+1,s] - (storages_liion.η_ch - .25*(1-m[:soh][nh,y,s]/m[:r_sto][a])) <=  m[:δbat][y,s])
 
end



function add_technical_constraints_my!(m::Model, grids::Vector{AbstractGrid}, nh::Int64, ny::Int64, ns::Int64)
    if !isempty(grids)
        na = length(grids)
        @constraints(m, begin
        # Power bounds
        [h in 1:nh, y in 1:ny, s in 1:ns, a in 1:na], m[:p_in][h,y,s,a]  <= grids[a].powerMax
        [h in 1:nh, y in 1:ny, s in 1:ns, a in 1:na], m[:p_out][h,y,s,a] <= grids[a].powerMax
        end)
    end
end


function add_technical_constraints_my!(m::Model, converters::Vector{AbstractConverter}, nh::Int64, ny::Int64, ns::Int64)
    if !isempty(converters)
        na = length(converters)
        @constraints(m, begin
        # Power bounds
        [h in 1:nh, y in 1:ny, s in 1:ns, a in 1:na], m[:p_c][h,y,s,a]  <= m[:r_c][a]
        end)
    end
end


# Power balance
function add_power_balance_my!(m::Model, mg::Microgrid, ω::AbstractScenarios, type::DataType, nh::Int64, ny::Int64, ns::Int64; ispnet::Bool=false)
    # !!! All the decision variables are defined positive !!!
    balance = AffExpr.(zeros(nh,ny,ns))
    # Demands and generation
    if !ispnet
        for (k,a) in enumerate(mg.demands)
            if a.carrier isa type
                add_to_expression!.(balance, ω.demands[k].power[:,:,:])
            end
        end
        # Generation
        for (k,a) in enumerate(mg.generations)
            if a.carrier isa type
                add_to_expression!.(balance, .- m[:r_g][k] .* ω.generations[k].power[:,:,:])
            end
        end
    else
        for (k,a) in enumerate(mg.demands)
            if a.carrier isa type
                add_to_expression!.(balance, m[:p_d][:,:,:,k])
            end
        end
        # Generation
        for (k,a) in enumerate(mg.generations)
            if a.carrier isa type
                add_to_expression!.(balance, .- m[:p_g][:,:,:,k])
            end
        end
    end
    # Storages
    for (k,a) in enumerate(mg.storages)
        if a.carrier isa type
            add_to_expression!.(balance, m[:p_ch][:,:,:,k] .- m[:p_dch][:,:,:,k])
        end
    end
    # Converters
    for (k,a) in enumerate(mg.converters)
        if type == Electricity
            if a isa Heater
                add_to_expression!.(balance, m[:p_c][:,:,:,k])
            elseif a isa Electrolyzer
                add_to_expression!.(balance, m[:p_c][:,:,:,k])
            elseif a isa AbstractFuelCell
                add_to_expression!.(balance, .- m[:p_c][:,:,:,k])
            end
        elseif type == Heat
            if a isa Heater
                add_to_expression!.(balance, .- m[:p_c][:,:,:,k] * a.η_E_H)
            elseif a isa Electrolyzer
                add_to_expression!.(balance, .- m[:p_c][:,:,:,k] * a.η_E_H)
            elseif a isa AbstractFuelCell
                add_to_expression!.(balance, .- m[:p_c][:,:,:,k] * a.η_H2_H / a.η_H2_E)
            end
        elseif type == Hydrogen
            if a isa Electrolyzer
                add_to_expression!.(balance, .- m[:p_c][:,:,:,k] * a.η_E_H2)
            elseif a isa AbstractFuelCell
                add_to_expression!.(balance, m[:p_c][:,:,:,k] / a.η_H2_E)
            end
        end
    end
    # Grids
    for (k,a) in enumerate(mg.grids)
        if a.carrier isa type
            add_to_expression!.(balance, .- m[:p_in][:,:,:,k] + m[:p_out][:,:,:,k])
        end
    end
    # Energy balance constraint
    if type == Electricity
        @constraint(m, electricity, balance .<= 0.)
    elseif type == Heat
        @constraint(m, heat, balance .<= 0.)
    elseif type == Hydrogen
        @constraint(m, hydrogen, balance .== 0.)
    end
end

# Renewable share
function add_renewable_share_my!(m::Model, mg::Microgrid, ω::AbstractScenarios, probabilities::Vector{Float64}, risk::AbstractRiskMeasure, nh::Int64, ny,::Int64, ns::Int64)
    total = zeros(ns)
    for (k,a) in enumerate(mg.demands)
        if a.carrier isa Electricity
            total .= total .+ sum(ω.demands[k].power[h,y,:] for h in 1:nh for y in 1:ny)
        elseif a.carrier isa Heat
            total .= total .+ sum(ω.demands[k].power[h,y,:] for h in 1:nh for y in 1:ny) ./ mg.converters[isin(mg.converters, Heater)[2]].η_E_H
        end
    end
    for (k,a) in enumerate(mg.grids)
        if a.carrier isa Electricity
            @expression(m, share[s in 1:ns], sum(m[:p_in][h,y,s,k] for h in 1:nh for y in 1:ny) - (1. - mg.parameters.renewable_share) * total[s])
        end
    end
    # Constraint according to CVaR
    @variables(m, begin
    ζ_s
    α_s[1:ns] >= 0.
    end)
    @constraints(m, begin
    [s in 1:ns], α_s[s] >= m[:share][s] - ζ_s
    ζ_s + 1 / (1 - beta(risk)) * sum(probabilities[s] * α_s[s] for s in 1:ns) <= 0.
    end)
end


# Objective
function add_design_objective!(m::Model, mg::Microgrid, ω::AbstractScenarios, nh::Int64, ns::Int64)
    # CAPEX
    capex = compute_capex(m, mg, ω)
    # OPEX
    opex = compute_opex(m, mg, ω, nh, ns)
    # Objective according to the CVaR
    
    
    @objective(m, Min, sum(capex + opex[s] for s in 1:ns))
end

# Objective
function add_design_objective_my!(m::Model, mg::Microgrid, ω::AbstractScenarios, probabilities::Vector{Float64}, risk::AbstractRiskMeasure, nh::Int64, ny::Int64, ns::Int64)
    # CAPEX
    capex = compute_capex_my(m, mg, ω, ny)
    # OPEX
    opex = compute_opex_my(m, mg, ω, nh, ny, ns)
    # Objective according to the CVaR
    
    
    @NLobjective(m, Min, sum(capex + opex[s] for s in 1:ns))
end



# Capex
function compute_capex_my(m::Model, mg::Microgrid, ω::AbstractScenarios, ny::Int64)
    cost = 0
    salvage = 0
    γ = repeat(1. ./ (1. + mg.parameters.τ) .^ range(0, length = mg.parameters.ny, step = mg.parameters.Δy), 1,1)


    # Generations
    for (k,a) in enumerate(mg.generations)

        salvage = salvage + γ[end] * ((a.lifetime - ny/a.lifetime)) * ω.generations[k].cost[1] * m[:r_g][k]
        cost = cost + ω.generations[k].cost[1] * m[:r_g][k]
    end
    # Storages 
    for (k,a) in enumerate(mg.storages)

        salvage = salvage + γ[end] * ω.storages[k].cost[1] * mean(m[:soh][end,ny,:])
        cost = cost + sum(γ[y] * m[:δbat][y] * ω.storages[k].cost[1] * m[:r_sto][k] for y in 1:ny)
    end
    # Converters
    for (k,a) in enumerate(mg.converters)

        salvage = salvage + γ[end] * ((a.lifetime - ny/a.lifetime)) * ω.converters[k].cost[1] * m[:r_c][k]
        cost = cost + ω.converters[k].cost[1] * m[:r_c][k]
    end

    return @NLexpression(m, cost - salvage)
end


# Grids
function compute_opex_my(m::Model, mg::Microgrid, ω::AbstractScenarios, nh::Int64, ny::Int64, ns::Int64)
    cost = AffExpr.(zeros(ns))
    for (k,a) in enumerate(mg.grids)
        add_to_expression!.(cost, sum((m[:p_in][h,y,:,k] .* ω.grids[k].cost_in[h,y,:] .- m[:p_out][h,y,:,k] .* ω.grids[k].cost_out[h,y,:]) .* mg.parameters.Δh  for h in 1:nh for y in 1:ny))
    end
    return cost
end
