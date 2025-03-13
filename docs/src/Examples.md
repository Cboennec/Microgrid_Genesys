# Examples


## Table of Contents

1. [Introduction](Examples.md#introduction)
2. [Generating Data Scenarios](Examples.md#generating-data-scenarios)
3. [Optimizing and Simulating the Microgrid](Examples.md#optimizing-and-simulating-the-microgrid)
4. [Extracting Metrics and Figures](Examples.md#extracting-metrics-and-figures)

## Introduction

This page provides example code snippets on how to use the main functionalities of our Julia package for modeling, optimizing and simulating microgrids. This package is designed to make it easy for users to design and analyze microgrid systems. The examples provided in this document cover the following key features:

- Generating data scenarios for simulation
- Optimizing and simulating the microgrid
- Extracting metrics and figures

## Generating Data Scenarios

In this section, we provide examples of how to generate data scenarios for the simulation of your microgrid system. These data scenarios will be used as input for the optimization and simulation processes.

We here present two ways of generating scenarios : 
A first way is to take directly the data from the Ausgrid dataset and for each client, each year create a scenario.  A second way is to build profils using a stochastic process detailed in [this article](https://hal.science/hal-03796717/document).

### Deterministic generation

```julia
 # Import necessary libraries
using JLD, Dates, Seaborn

# Include supplementary functions
include("functions.jl")

# Define global constants
# nh: number of hours in a year
# ny: number of years to simulate
# ns: number of scenarios to generate
const nh, ny, ns = 8760, 2, 60

# Load data for Ausgrid network for the years 2010, 2011, 2012
data_2010 = load(joinpath("Examples","build_scenarios", "data", "clean_dataset_ausgrid_2010_2011.jld"), "clean_dataset")
data_2011 = load(joinpath("Examples","build_scenarios", "data", "clean_dataset_ausgrid_2011_2012.jld"), "clean_dataset")
data_2012 = load(joinpath("Examples","build_scenarios", "data", "clean_dataset_ausgrid_2012_2013.jld"), "clean_dataset")

# Identify customers who have data for all three years
customers = [k for k in keys(data_2010) if k in keys(data_2011) && k in keys(data_2012)]

# Initialize empty arrays for Photovoltaic (PV) production, Electric Load (ld_E) and Heat Load (ld_H)
_pv, _ld_E, _ld_H = [], [], []

# Set up date format to be used
dateformat = Dates.DateFormat("dd-u-yyyy")

# Loop over each customer and collect their data for all three years
for c in customers[1:20]
    # Collect and format data for Electric Load
    push!(_ld_E, (t = hcat(Dates.DateTime.(data_2010[c]["GC"]["time"][1:nh], dateformat), Dates.DateTime.(data_2011[c]["GC"]["time"][1:nh], dateformat), Dates.DateTime.(data_2012[c]["GC"]["time"][1:nh], dateformat)),
            power = hcat(data_2010[c]["GC"]["power"][1:nh], data_2011[c]["GC"]["power"][1:nh], data_2012[c]["GC"]["power"][1:nh])))

    # Collect and format data for Heat Load
    push!(_ld_H, (t = hcat(Dates.DateTime.(data_2010[c]["CL"]["time"][1:nh], dateformat), Dates.DateTime.(data_2011[c]["CL"]["time"][1:nh], dateformat), Dates.DateTime.(data_2012[c]["CL"]["time"][1:nh], dateformat)),
            power = hcat(data_2010[c]["CL"]["power"][1:nh], data_2011[c]["CL"]["power"][1:nh], data_2012[c]["CL"]["power"][1:nh])))

    # Collect and format data for PV Production
    push!(_pv, (t = hcat(Dates.DateTime.(data_2010[c]["GG"]["time"][1:nh], dateformat), Dates.DateTime.(data_2011[c]["GG"]["time"][1:nh], dateformat), Dates.DateTime.(data_2012[c]["GG"]["time"][1:nh], dateformat)),
            power = hcat(data_2010[c]["GG"]["power"][1:nh], data_2011[c]["GG"]["power"][1:nh], data_2012[c]["GG"]["power"][1:nh])))
end

# Format data to be compatible with Genesys simulation system
timestamp = repeat(reshape(hcat([p.tfor p in _pv]...), nh, 1, ns), 1, ny, 1)
pv = repeat(reshape(hcat([p.power for p in _pv]...), nh, 1, ns), 1, ny, 1)
ld_E = repeat(reshape(hcat([ld.power for ld in _ld_E]...), nh, 1, ns), 1, ny, 1)
ld_H = repeat(reshape(hcat([ld.power for ld in _ld_H]...), nh, 1, ns), 1, ny, 1)

# Define scenarios with costs
ω = Dict(
    "pv" => (t = timestamp, power = pv, cost = 1300 * ones(ny, ns)),  # Scenario for PV with timestamp, power and cost
    "ld_E" => (t = timestamp, power = ld_E),  # Scenario for Electric Load with timestamp and power
    "ld_H" => (t = timestamp, power = ld_H),  # Scenario for Heat Load with timestamp and power
    "liion" => (cost = 300 * ones(ny, ns),),  # Scenario for Lithium-ion batteries with cost
    "tes" => (cost = 10 * ones(ny, ns),),  # Scenario for Thermal Energy Storage with cost
    "h2tank" => (cost = 10 * ones(ny, ns),),  # Scenario for Hydrogen Storage Tank with cost
    "elyz" => (cost = 1300 * ones(ny, ns),),  # Scenario for Electrolyzer with cost
    "fc" => (cost = 1700 * ones(ny, ns),),  # Scenario for Fuel Cell with cost
    "heater" => (cost = 0 * ones(ny, ns),),  # Scenario for Heater with cost
    "grid" => (cost_in = 0.19 * ones(nh, ny, ns), cost_out = 0.0001 * ones(nh, ny, ns)),  # Scenario for Grid with cost_in and cost_out
)

# Save scenarios as .jld files
save(joinpath("data", "ausgrid_deterministic.jld"), "ω", ω)
```

### Stochastic generation

```julia
include("..\\src\\Genesys.jl")

using Main.Genesys
using Distributions, CSV, JLD, Dates, Seaborn, Statistics, ProgressMeter

# Global constants
const nh, ny, ns = 8760, 1, 2000

# Load data related to multiple clients over 3 years
data_2010 = load(joinpath("stage_scenario", "data", "clean_dataset_ausgrid_2010_2011.jld"), "clean_dataset")
data_2011 = load(joinpath("stage_scenario", "data", "clean_dataset_ausgrid_2011_2012.jld"), "clean_dataset")
data_2012 = load(joinpath("stage_scenario", "data", "clean_dataset_ausgrid_2012_2013.jld"), "clean_dataset")

# Find customers with 3 years of data
customers = [k for k in keys(data_2010) if k in keys(data_2011) && k in keys(data_2012)]

# Initialize
dateformat = Dates.DateFormat("dd-u-yyyy")
_pv, _ld_E, _ld_H = [], [], []
s0 = [0., 0., 0.]
t0 = DateTime(2020,7,1,0)

# For each customer with 3 years of data, we use the data to create three time series (photovoltaic production, electric consumption, heat consumption)
# These time series are used to construct the states and probabilities of the Markov chains using the "generator"
# We then use this generator to produce a set of ns scenarios at an hourly timestep using "generate"
# The generated scenarios are stored in three lists (_pv, _ld_E, _ld_H)

@showprogress for c in custumers[1:20]
        # Retrieve the scenario
        ld_E = (t = hcat(Dates.DateTime.(data_2010[c]["GC"]["time"][1:nh], dateformat), Dates.DateTime.(data_2011[c]["GC"]["time"][1:nh], dateformat), Dates.DateTime.(data_2012[c]["GC"]["time"][1:nh], dateformat)),
                power = hcat(data_2010[c]["GC"]["power"][1:nh], data_2011[c]["GC"]["power"][1:nh], data_2012[c]["GC"]["power"][1:nh]))

        ld_H = (t = hcat(Dates.DateTime.(data_2010[c]["CL"]["time"][1:nh], dateformat), Dates.DateTime.(data_2011[c]["CL"]["time"][1:nh], dateformat), Dates.DateTime.(data_2012[c]["CL"]["time"][1:nh], dateformat)),
                power = hcat(data_2010[c]["CL"]["power"][1:nh], data_2011[c]["CL"]["power"][1:nh], data_2012[c]["CL"]["power"][1:nh]))

        pv = (t = hcat(Dates.DateTime.(data_2010[c]["GG"]["time"][1:nh], dateformat), Dates.DateTime.(data_2011[c]["GG"]["time"][1:nh], dateformat), Dates.DateTime.(data_2012[c]["GG"]["time"][1:nh], dateformat)),
                power = hcat(data_2010[c]["GG"]["power"][1:nh], data_2011[c]["GG"]["power"][1:nh], data_2012[c]["GG"]["power"][1:nh]))

        # Initialize generator
        generator = Genesys.initialize_generator!(MarkovGenerator(nstate = 20, algo = "kmeans"), pv, ld_E, ld_H)

        # Generate
        ω, proba = Genesys.generate(generator, s0, t0, nh, ny = ny, ns = ns)

        # Store
        push!(_pv, ω[1])
        push!(_ld_E, ω[2])
        push!(_ld_H, ω[3])
end

# After generating these time series, we will create scenarios for our case study
# First by aggregating the consumption data to create a case of 5 consumers
# Then by adding certain elements including pricing which will be necessary for our metrics
# Finally, we will store all these elements in two dictionaries, one for sizing and one for analysis (resp. optim, simu)
# These dictionaries will then be saved under the labels ω_optim and ω_simu


# Aggregation
# Randomly chose 5 among the 20 availables
idx = rand(1:20, 5, ns)
# Aggregate except for PV
pv = _pv[1]
ld_E = reshape(hcat([sum(ld[:,:,s] for ld in _ld_E[idx[:,s]]) for s in 1:ns]...), nh, ny, ns)
ld_H = reshape(hcat([sum(ld[:,:,s] for ld in _ld_H[idx[:,s]]) for s in 1:ns]...), nh, ny, ns)

# Add first year
timestamp = repeat(t0:Hour(1):t0+Hour(nh-1), 1, 2, ns)
pv = cat(pv[:,1,1] .* ones(nh,1,ns), pv, dims=2)
ld_E = cat(ld_E[:,1,1] .* ones(nh,1,ns), ld_E, dims=2)
ld_H = cat(ld_H[:,1,1] .* ones(nh,1,ns), ld_H, dims=2)

# Operating cost
# Flat rate including taxes, etc. - from eurostat - projected evolution over 20 years
_cost_in = compute_operating_cost_scenarios(0.19 * ones(nh), Normal(1.04, 0.04), nh, 20, ns)
_cost_out = compute_operating_cost_scenarios(0.0001 * ones(nh), Normal(1., 0.), nh, 20, ns)
# Averaged cost
cost_in = mean(_cost_in, dims=2)
cost_out = mean(_cost_out, dims=2)
# Add first year
cost_in = cat(0.19 * ones(nh, 1, ns), cost_in, dims=2)
cost_out = cat(0.0001 * ones(nh, 1, ns), cost_out, dims=2)

# Investment cost - from Petkov & Gabrielli, 2020
cost_pv = 1300 * ones(2, ns)
cost_liion = 300 * ones(2, ns)
cost_tes = 10 * ones(2, ns)
cost_h2tank = 10 * ones(2, ns)
cost_elyz = 1300 * ones(2, ns)
cost_fc = 1700 * ones(2, ns)
cost_heater = 0 * ones(2, ns)

# Here we can devide the set of scenario in two keeping one half for sizing and one for assessing
# Scenarios
ω_optim = Dict(
"pv" => (t = timestamp[:,:,2:2:end], power = pv[:,:,2:2:end], cost = cost_pv[:,2:2:end]),
"ld_E" => (t = timestamp[:,:,2:2:end], power = ld_E[:,:,2:2:end]),
"ld_H" => (t = timestamp[:,:,2:2:end], power = ld_H[:,:,2:2:end]),
"liion" => (cost = cost_liion[:,2:2:end],),
"tes" => (cost = cost_tes[:,2:2:end],),
"h2tank" => (cost = cost_h2tank[:,2:2:end],),
"elyz" => (cost = cost_elyz[:,2:2:end],),
"fc" => (cost = cost_fc[:,2:2:end],),
"heater" => (cost = cost_heater[:,2:2:end],),
"grid" => (cost_in = cost_in[:,:,2:2:end], cost_out = cost_out[:,:,2:2:end]),
)

# ... and odd number scenarios for simulation
ω_simu = Dict(
"pv" => (t = timestamp[:,:,1:2:end], power = pv[:,:,1:2:end], cost = cost_pv[:,1:2:end]),
"ld_E" => (t = timestamp[:,:,1:2:end], power = ld_E[:,:,1:2:end]),
"ld_H" => (t = timestamp[:,:,1:2:end], power = ld_H[:,:,1:2:end]),
"liion" => (cost = cost_liion[:,1:2:end],),
"tes" => (cost = cost_tes[:,1:2:end],),
"h2tank" => (cost = cost_h2tank[:,1:2:end],),
"elyz" => (cost = cost_elyz[:,1:2:end],),
"fc" => (cost = cost_fc[:,1:2:end],),
"heater" => (cost = cost_heater[:,1:2:end],),
"grid" => (cost_in = cost_in[:,:,1:2:end], cost_out = cost_out[:,:,1:2:end]),
)

# Save scenarios as .jld files
save(joinpath("data", "ausgrid_5_twostage.jld"), "ω_optim", ω_optim, "ω_simu", ω_simu)
```

For more information on  `initialize_generator` and  `generate` see [Scenario page](documentation.md)
     


## Optimizing and Simulating the Microgrid

Once you have generated the data scenarios for your microgrid system, you can optimize and simulate its performance using the functions provided in our package. In this section, we demonstrate how to perform optimization and simulation using the data scenarios created in the previous section.


### Constructing the grid
The first step is to build the microgrid get the data and add assets to it. The following code show this process for the case of a single energy grid with battery storage and Solar panel over 21 years for 1 scenario. For more details about the assets see [Assets page](documentation.md)

```julia

using JLD, Dates

const nh, ny, ns = 8760, 21, 1 #nh = number of timestep per year, ny = time horizon (in years), ns = number of scenario used

data = load(joinpath("example","data","ausgrid_5_twostage.jld")) #Get the data for scenarios 

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = 1.)) # Instantiate a Microgrid 


add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion_electro_chimique(),
                Grid(carrier = Electricity())) # Add assets to it

ω_d = Scenarios(microgrid, data["ω_optim"]) # Build the right number of scenarios with the right length for the microgrid

```

It is also possible to get 2 set of scenarios as explained in section [Stochastic generation](#stochastic-generation) of this page. One for design and one for assessment.
```julia
ω_d, ω_a = Scenarios(microgrid, data["ω_optim"]), Scenarios(microgrid, data["ω_simu"])
```

### Sizing the grid

The sizing is performed by a component called `designer`. Several type of designer are implemented and their role is to determine the size of the asset. The simplest designer is the manual one where the user chose mannually the size of the assets. Other designer use optimization to provide a size, different optimization methods are implemented, exact and stochastic ones.

Considering a grid `mg`, in order to determine the size of its assets, 

one can use it's own design and use a `Manual` design.

```julia
PV_size = 20.
BAT_size = 40.                
designer = initialize_designer!(mg, Manual(generations = [PV_size], storages = [BAT_size], subscribed_power = [5.]), ω_d)
```
`Manual` design require a vector of values for the size of the generation assets, storage assets and grid assets.

Other design methods can be used with the following code. Here an example for the MILP designer

```julia
initialize_designer!(microgrid, MILP(options = MILPOptions(reducer = ManualReducer(y=1:1, s=1:1))), ω_d)
```
Please note that the models used in the MILP formulation are linear models. Even if you added a more fine battery model.

### Operating the grid
To make the decisions during the simulation the grid need to be controlled. This role is taken by the `Controller`. A controller is going to operate the grid during the simulation, their is several type of controller implemented (Need to add reference or details) here. The simplest ones are Rule Based Controller `RBC`.  

```julia
controller = initialize_controller!(microgrid,  RBC(options = RBCOptions(policy_selection = 2 )), ω_d)
```

More complexe controller using optimisation with varying range of sight can also be used.

```julia
controller = initialize_controller!(microgrid, Anticipative(generations = [10.], storages = [20.], converters = []), designer, ω_d)
```

Please note that the models used in the MILP formulation are linear models. Even if you added a more fine battery model.

## Simulating 

Now that everything is declared the microgrid can be simulated to assess techno-economic indicators.

```julia
@time simulate!(microgrid, controller, designer, ω_a, options = Genesys.Options(mode = "serial", firstyear = false))
```
Here the microgrid will be simulated using the sizing stored in `desginer`, the operation stored in `controller` under the conditions of `ω_a`.
The first year option can be use to compare a first year without any asset installed (as some kind of reference) to another year. 



## Extracting Metrics and Figures
After optimizing and simulating your microgrid system, you may want to analyze its performance by extracting relevant metrics and creating visualizations. In this section, we provide examples of how to use our package to generate these metrics and figures.


```julia
 #Example code: Generating Metrics and Figures
```



Once you have completed the examples in this document, you should be familiar with the main functionalities of our Julia package for modeling, optimizing, and simulating microgrids. If you have any questions or need further assistance, please refer to our package documentation or contact us for support.