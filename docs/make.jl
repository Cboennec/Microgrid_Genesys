using Pkg
Pkg.add(url="https://github.com/hradet/Metaheuristics.jl.git")
using Metaheuristics

Pkg.activate(".")

include("..\\src\\Genesys.jl")

using Documenter
using Main.Genesys

makedocs(sitename="Microgrid Genesys",
          modules=[Genesys],
          format=Documenter.HTML())



