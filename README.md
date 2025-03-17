# Genesys

A generic module written in Julia to asses and compare different design and control methods for distributed energy systems in a stochastic framework. The `simulate!` function includes multi-stage investment periods and multi-scenarios assessment.  


# Installation
In order to use the package, follow the [managing package guideline](https://julialang.github.io/Pkg.jl/v1/managing-packages/) for uneregistred packages. Examples on how to use the package are provided in the "example" folder. 

If not executed as a package, simply use `include(joinpath(pwd(),"src","Genesys2.jl"))` to execute all the package files.


# Features 

The package is built for modularity and allows several modeling and resolution choices. 

## Component Models
- Lithium Ion batteries
  - 2 energy models
  - 4 aging models
- Electrolyzers
  - 3 energy models
  - 2 aging models
- Fuel Cells
  - 3 energy models
  - 3 aging models
- Thermal Storages
- Solar Panel
- Hydrogen Tank

## Resolutions methods
- Design
  - Manual
  - MILP 
  - Metaheuristic
 
- Control
  - Dummy
  - Anticipative
  - Rule based (RBC)
  - Open Loop Feedback Control (OLFC) - OLFC with a single scenario is equivalent to MPC...
 
 ## Scenarios generation and reduction methods
- Generation  
  - Anticipative
  - Markov

- Reduction
  - Manual
  - SAA
  - Mean value
  - Feature based


# Documentation 
Full documentation, examples, and tutorial (in french) are available [here](https://cboennec.github.io/Microgrid_Genesys/)
