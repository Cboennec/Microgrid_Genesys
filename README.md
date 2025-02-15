# Genesys

A generic module written in Julia to asses and compare different design and control methods for distributed energy systems in a stochastic framework. The `simulate!` function includes multi-stage investment periods and multi-scenarios assessment.  

# Branch
v2 branch is now this offical branch
=======
The current maintained and up-to-date branch is not "main" anymore but "v2".


# Installation
In order to use the package, follow the [managing package guideline](https://julialang.github.io/Pkg.jl/v1/managing-packages/) for uneregistred packages. Examples on how to use the package are provided in the "example" folder. 

# Resolutions methods
- Design
  - Manual
  - MILP 
  - Metaheuristic
 
- Control
  - Dummy
  - Anticipative
  - Rule based (RBC)
  - Open Loop Feedback Control (OLFC) - OLFC with a single scenario is equivalent to MPC...
 
 # Scenario generation and reduction methods
- Generation  
  - Anticipative
  - Markov

- Reduction
  - Manual
  - SAA
  - Mean value
  - Feature based


# Documentation 
Full documentation (under construction) is available [here](https://cboennec.github.io/Microgrid_Genesys/)
