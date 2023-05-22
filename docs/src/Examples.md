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

```julia
 #Example code: Generating data scenarios
```


## Optimizing and Simulating the Microgrid

Once you have generated the data scenarios for your microgrid system, you can optimize and simulate its performance using the functions provided in our package. In this section, we demonstrate how to perform optimization and simulation using the data scenarios created in the previous section.


### Constructing the grid
The first step is to build the microgrid get the data and add assets to it. The following code show this process for the case of a single energy grid with battery storage and Solar panel over 21 years for 1 scenario. For more details about the assets see [Assets page](assets.md)

```julia

using JLD, Dates

const nh, ny, ns = 8760, 21, 1 #nh = number of timestep per year, ny = time horizon (in years), ns = number of scenario used

data = load(joinpath("example","data","ausgrid_5_twostage.jld")) #Get the data for scenarios 

mg = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = 1.)) # Instantiate a Microgrid 


add!(mg, Demand(carrier = Electricity()),
                Solar(),
                Liion_electro_chimique(),
                Grid(carrier = Electricity())) # Add assets to it

ω_d = Scenarios(mg, data["ω_optim"]) # Build the right number of scenarios with the right length for the microgrid

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

Other design methods can be used with the following code.

```julia
#Code for using a opt desginer
```


### Operating the grid
To make the decisions during the simulation the grid need to be controlled. This role is taken by the `Controller`. A controller is going to operate the grid during the simulation, their is several type of controller implemented (Need to add reference or details) here. The simplest ones are Rule Based Controller `RBC`.  

```julia
controller = initialize_controller!(mg,  RBC(options = RBCOptions(policy_selection = 2 )), ω_d)
```

More complexe controller using optimisation with varying range of sight can also be used.

```julia
#Code for using a opt controller
```


## Extracting Metrics and Figures
After optimizing and simulating your microgrid system, you may want to analyze its performance by extracting relevant metrics and creating visualizations. In this section, we provide examples of how to use our package to generate these metrics and figures.


```julia
 #Example code: Generating Metrics and Figures
```



Once you have completed the examples in this document, you should be familiar with the main functionalities of our Julia package for modeling, optimizing, and simulating microgrids. If you have any questions or need further assistance, please refer to our package documentation or contact us for support.