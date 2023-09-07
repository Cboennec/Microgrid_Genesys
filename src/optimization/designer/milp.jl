#=
    Designer based on the equivalent annual cost (EAC) with multiple scenarios
=#


mutable struct MILPOptions
  exact::Bool
  solver::Module
  reducer::AbstractScenariosReducer
  objective_risk::AbstractRiskMeasure
  share_risk::AbstractRiskMeasure
  reopt::Bool
  read_reduction::Union{String, Nothing}
  write_reduction::Union{String, Nothing}


  MILPOptions(; exact = true,
                solver = Gurobi,
                reducer = FeatureBasedReducer(),
                objective_risk = Expectation(),
                share_risk = Expectation(),
                reopt = false,
                read_reduction = nothing,
                write_reduction = nothing) =
                new(exact, solver, reducer, objective_risk, share_risk, reopt, read_reduction, write_reduction)
end

mutable struct MILP <: AbstractDesigner
    options::MILPOptions
    decisions::NamedTuple
    model::JuMP.Model
    history::AbstractScenarios

    MILP(; options = MILPOptions()) = new(options)
end

### Models
function build_model(mg::Microgrid, designer::MILP, ω::Scenarios, probabilities::Vector{Float64})
    # Sets
    nh, ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 3)
    # Initialize
    m = Model(designer.options.solver.Optimizer)
    # Add desgin decision variables
    add_investment_decisions!(m, mg.generations)
    add_investment_decisions!(m, mg.storages)
    add_investment_decisions!(m, mg.converters)
    # Add operation decision variables
    add_operation_decisions!(m, mg.storages, nh, ns)
    add_operation_decisions!(m, mg.converters, nh, ns)
    add_operation_decisions!(m, mg.grids, nh, ns)
    # Add design constraints
    add_investment_constraints!(m, mg.generations)
    add_investment_constraints!(m, mg.storages)
    add_investment_constraints!(m, mg.converters)
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
    # Renewable share constraint
    add_renewable_share!(m, mg, ω, probabilities, designer.options.share_risk, nh, ns)
    # Objective
    add_design_objective!(m, mg, ω, probabilities, designer.options.objective_risk, nh, ns)
    return m
end


### Models
function build_model_robust(mg::Microgrid, designer::MILP, ω::Scenarios)
    # Sets
    nh,ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 3)
    # Initialize
    m = Model(designer.options.solver.Optimizer)
    set_attribute(m, "OutputFlag", 0)
    # Add desgin decision variables
    add_investment_decisions!(m, mg.generations)
    add_investment_decisions!(m, mg.storages)
    add_investment_decisions!(m, mg.converters)
    # Add operation decision variables
    add_operation_decisions!(m, mg.storages, nh, ns)
    add_operation_decisions!(m, mg.converters, nh, ns)
    add_operation_decisions!(m, mg.grids, nh, ns)
    # Add design constraints
    add_investment_constraints!(m, mg.generations)
    add_investment_constraints!(m, mg.storages)
    add_investment_constraints!(m, mg.converters)
    # Add technical constraints
    add_technical_constraints!(m, mg.storages, mg.parameters.Δh, nh, ns)
    add_technical_constraints!(m, mg.converters, nh, ns)
    add_technical_constraints!(m, mg.grids, nh, ns)
    # Add periodicity constraint
    #add_periodicity_constraints!(m, mg.storages, ns)
    # Add power balance constraints
    add_power_balance!(m, mg, ω, Electricity, nh, ns)
    add_power_balance!(m, mg, ω, Heat, nh, ns)
    add_power_balance!(m, mg, ω, Hydrogen, nh, ns)
    # Renewable share constraint
    add_renewable_share_robust!(m, mg, ω, nh, ns)
    # Objective
    add_design_objective!(m, mg, ω, nh, ns)
    return m
end

### Offline
function initialize_designer!(mg::Microgrid, designer::MILP, ω::Scenarios; multiyear::Bool=false, robust::Bool=false)
    # Preallocate
    preallocate!(mg, designer)

    if mg.parameters.ns > 1

        # Scenario reduction from the optimization scenario pool
        if isa(designer.options.read_reduction, Nothing)
            println("Starting scenario reduction...")
            ω_reduced, probabilities = reduce(designer.options.reducer, ω)
            # Saving
            if !isa(designer.options.write_reduction, Nothing)
                JLD.save(designer.options.write_reduction, "scenarios", ω_reduced, "probabilities", probabilities)
            end
        else
            println("Reading scenario reduction from file...")
            ω_reduced = load(designer.options.read_reduction, "scenarios")
            probabilities = load(designer.options.read_reduction, "probabilities")
        end

    else
        ω_reduced = ω
        probabilities = 1
    end
    
    # Initialize model
    println("Building the model...")
    if robust
        println("robust")
        designer.model = build_model_robust(mg, designer, ω_reduced)
    elseif multiyear
        println("Multi_year")
        designer.model = build_model_multi_years(mg, designer, ω_reduced, probabilities)
    else
        println("my")
        designer.model = build_model_my(mg, designer, ω_reduced, probabilities)
    end


    # Compute investment decisions for the first year
    println("Starting optimization...")
    optimize!(designer.model)

    if multiyear
        # Assign values
        for k in 1:length(mg.generations)
            designer.decisions.generations[k][1,:] .= value(designer.model[:r_g][k])
        end
        for k in 1:length(mg.storages)
            for y in 1:mg.parameters.ny
                designer.decisions.storages[k][y,:] .= value(designer.model[:r_sto][y,k])
            end
        end
        for k in 1:length(mg.converters)
            designer.decisions.converters[k][1,:] .= value(designer.model[:r_c][k])
        end
    else
        # Assign values
        for k in 1:length(mg.generations)
            designer.decisions.generations[k][1,:] .= value(designer.model[:r_g][k])
        end
        for k in 1:length(mg.storages)
            designer.decisions.storages[k][1,:] .= value(designer.model[:r_sto][k])
        end
        for k in 1:length(mg.converters)
            designer.decisions.converters[k][1,:] .= value(designer.model[:r_c][k])
        end
    end


    # Save history
    designer.history = ω_reduced

     return designer
end

### Online
function compute_investment_decisions!(y::Int64, s::Int64, mg::Microgrid, designer::MILP)
    # TODO
end

### Utils
beta(risk::WorstCase) = 1. - 1e-6
beta(risk::Expectation) = 0.
beta(risk::CVaR) = risk.β
Γ(τ::Float64, lifetime::Union{Float64, Int64}) = τ * (τ + 1.) ^ lifetime / ((τ + 1.) ^ lifetime - 1.)






### Models
function build_model_my(mg::Microgrid, designer::MILP, ω::Scenarios, probabilities::Vector{Float64})
    # Sets
    nh, ny, ns = size(ω.demands[1].power, 1), size(ω.demands[1].power, 2), size(ω.demands[1].power, 3)
    # solver
    
    solver = designer.options.exact ? Alpine : Juniper
    nl_option_name = designer.options.exact ? "nlp_solver" : "nl_solver"
    
    gurobi = optimizer_with_attributes(Gurobi.Optimizer, 
                                         MOI.Silent() => true,
                                         "Presolve"   => 1) 
    highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0)

    mip_solver = designer.options.exact ? gurobi : highs

    optimizer = optimizer_with_attributes(solver.Optimizer,
    
                            nl_option_name => ipopt,
                            "mip_solver" => gurobi,#mip_solver
                            "time_limit" => 600)

                   

    # Add desgin decision variables
    add_investment_decisions!(m, mg.generations)
    add_investment_decisions!(m, mg.storages)
    add_investment_decisions!(m, mg.converters)
    # Add operation decision variables
    add_operation_decisions_my!(m, mg.storages, mg.generations, mg.converters, mg.grids, nh, ny, ns)

    # Add design constraints
    add_investment_constraints!(m, mg.generations)
    add_investment_constraints!(m, mg.storages)
    add_investment_constraints!(m, mg.converters)
    # Add technical constraints
    add_technical_constraints_my!(m, mg.storages, mg.parameters.Δh, nh, ny, ns)
    add_technical_constraints_my!(m, mg.grids, nh, ny, ns)
    add_technical_constraints_my!(m, mg.converters, nh, ny, ns)

    # Add periodicity constraint
    add_coupling_constraints_my!(m, mg.storages, nh, ny, ns)

    # Add power balance constraints
    add_power_balance_my!(m, mg, ω, Electricity, nh, ny, ns)
    add_power_balance_my!(m, mg, ω, Heat, nh, ny, ns)
    add_power_balance_my!(m, mg, ω, EnergyCarrier, nh, ny, ns)
    # Renewable share constraint

    add_renewable_share_robust!(m, mg, ω, probabilities, designer.options.share_risk, nh, ny, ns)
    # Objective
    add_design_objective_my!(m, mg, ω, probabilities, designer.options.objective_risk, nh, ny, ns)
    return m
end



### Offline
function initialize_designer_my!(mg::Microgrid, designer::MILP, ω::Scenarios; multiyear::Bool=false)
    # Preallocate
    preallocate!(mg, designer)

    # Scenario reduction from the optimization scenario pool
    if isa(designer.options.read_reduction, Nothing)
        println("Starting scenario reduction...")
        ω_reduced, probabilities = reduce(designer.options.reducer, ω)
        # Saving
        if !isa(designer.options.write_reduction, Nothing)
            JLD.save(designer.options.write_reduction, "scenarios", ω_reduced, "probabilities", probabilities)
        end
    else
        println("Reading scenario reduction from file...")
        ω_reduced = load(designer.options.read_reduction, "scenarios")
        probabilities = load(designer.options.read_reduction, "probabilities")
    end

    # Initialize model
    if multiyear
        designer.model = build_model_multi_years(mg, designer, ω_reduced, probabilities)
    else
        println("Building the model...")
        designer.model = build_model_my(mg, designer, ω_reduced, probabilities)
    end


    # Compute investment decisions for the first year
    println("Starting optimization...")
    optimize!(designer.model)

    if multiyear
        # Assign values
        for k in 1:length(mg.generations)
            designer.decisions.generations[k][1,:] .= value(designer.model[:r_g][k])
        end
        for k in 1:length(mg.storages)
            for y in 1:mg.parameters.ny
                designer.decisions.storages[k][y,:] .= value(designer.model[:r_sto][y,k])
            end
        end
        for k in 1:length(mg.converters)
            designer.decisions.converters[k][1,:] .= value(designer.model[:r_c][k])
        end
    else
        # Assign values
        for k in 1:length(mg.generations)
            designer.decisions.generations[k][1,:] .= value(designer.model[:r_g][k])
        end
        for k in 1:length(mg.storages)
            designer.decisions.storages[k][1,:] .= value(designer.model[:r_sto][k])
        end
        for k in 1:length(mg.converters)
            designer.decisions.converters[k][1,:] .= value(designer.model[:r_c][k])
        end
    end


    # Save history
    designer.history = ω_reduced

    return designer
end
