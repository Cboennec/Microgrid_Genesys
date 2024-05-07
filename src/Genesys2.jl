#This code generate the Genesys module grouping :
# -Asset models
# -Data Models
# -Optimization features for design and operation
# -Simulation code
# -Analysis tools including metrics and display tools



#module Genesys
# TODO: break into submodules

#Abstract (assets and operation/design) types for adaptable code
abstract type AbstractDemand end
abstract type AbstractGeneration end
abstract type AbstractStorage end
abstract type AbstractConverter end
abstract type AbstractGrid end

abstract type AbstractDesigner end
abstract type AbstractDesignerNonFloat end
abstract type AbstractController end

abstract type MetaheuristicResults end


abstract type EnergyCarrier end

#Metaheuristic module part
using Distributions, Random, LinearAlgebra, Statistics
using Distributed
using GlobalSensitivity, Test, ProgressMeter

# Clearing
include(joinpath("optimization","designer", "metaheuristic","clearing","struct.jl"))
include(joinpath("optimization","designer", "metaheuristic","clearing","functions.jl"))
include(joinpath("optimization","designer", "metaheuristic","clearing","optimize.jl"))
include(joinpath("optimization","designer", "metaheuristic","clearing","utils.jl"))
# NSGAII
include(joinpath("optimization","designer", "metaheuristic","NSGAII","struct.jl"))
include(joinpath("optimization","designer", "metaheuristic","NSGAII","functions.jl"))
include(joinpath("optimization","designer", "metaheuristic","NSGAII","optimize.jl"))
include(joinpath("optimization","designer", "metaheuristic","NSGAII","utils.jl"))

#Genesys package part 


# Optimisation
using JuMP, Cbc, Metaheuristics, SDDP, Ipopt#, Gurobi
# Math
using Statistics, StatsBase, MultivariateStats, Clustering, Distributions, Distances, LinearAlgebra, Interpolations
# Others
using Seaborn, Dates, Distributed, SharedArrays, CSV, DataFrames, JLD2, Pandas, Random, Plots
# Assets
include(joinpath("assets","microgrid.jl"))
include(joinpath("assets","carriers.jl"))
include(joinpath("assets","liion","liion.jl"))
include(joinpath("assets","tes.jl"))
include(joinpath("assets","h2tank.jl"))
include(joinpath("assets","electrolyzer","electrolyzer.jl"))
include(joinpath("assets","fuelcell","fuelcell.jl"))
include(joinpath("assets","heater.jl"))
include(joinpath("assets","grid.jl"))
include(joinpath("assets","solar.jl"))
include(joinpath("assets","demand.jl"))
#export AbstractController,  AbstractDesigner
#export AbstractLiion, AbstractFuelCell
#export Microgrid, Demand, Solar
#export Liion_energy_exchanged, Liion_rainflow, Liion_fixed_lifetime, Liion_vermeer, Liion_electro_chimique, Tremblay_dessaint_params, vermeer_params, Electro_chimique_params
#export ThermalStorage, H2Tank, FuelCell_OnOFF, FuelCell_HoursMax, FuelCell_Power, Electrolyzer, Heater, Grid, GlobalParameters
#export Electricity, Heat, Hydrogen
#export add!
# Scenarios
include(joinpath("scenarios","scenarios.jl"))
include(joinpath("scenarios","reduction.jl"))
include(joinpath("scenarios","generation.jl"))
include(joinpath("scenarios","utils.jl"))
#export Scenarios, MiniScenarios
#export ManualReducer, SAAReducer, MeanValueReducer, FeatureBasedReducer
#export UnitRangeTransform, ZScoreTransform
#export PCAReduction, StatsReduction
#export KmedoidsClustering
#export MarkovGenerator, AnticipativeGenerator
#export reduce, generate
# Optimization utils
include(joinpath("optimization","utils.jl"))
#export Expectation, CVaR, WorstCase
# Operation optimization
include(joinpath("optimization","controller","dummy.jl"))
include(joinpath("optimization","controller","anticipative.jl"))
include(joinpath("optimization","controller","rb.jl"))
include(joinpath("optimization","controller","olfc.jl"))
include(joinpath("optimization","controller","multi_year_anticipative.jl"))
#export Dummy, RBC, Anticipative, OLFC, Multi_Year_Anticipative
#export RBCOptions, AnticipativeOptions, OLFCOptions, Multi_Year_Anticipative_Options
#export initialize_controller!
# Investment optimization
include(joinpath("optimization","designer","manual.jl"))
include(joinpath("optimization","designer","milp.jl"))
include(joinpath("optimization","designer","metaheuristic", "metaheuristic.jl"))
#export Manual, Metaheuristic, MILP
#export MetaheuristicOptions, MILPOptions, ManualOptions
#export initialize_designer!, initialize_designer_my!
# Simulation
include(joinpath("simulation","informations.jl"))
include(joinpath("simulation","dynamics.jl"))
include(joinpath("simulation","power_balances.jl"))
include(joinpath("simulation","simulations.jl"))
#export simulate!
# Utils
include(joinpath("utils","metrics.jl"))
include(joinpath("utils","plots.jl"))
include(joinpath("utils","saves.jl"))
#export Metrics
#export plot_operation, plot_metrics
#export COST

#end


