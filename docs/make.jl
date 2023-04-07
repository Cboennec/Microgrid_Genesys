using Pkg
Pkg.activate(".")

include("..\\src\\Genesys.jl")

using Documenter
using Main.Genesys

makedocs(sitename="Microgrid Genesys",
          modules=[Genesys],
          format=Documenter.HTML())



