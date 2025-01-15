
import Pkg
using Pkg

ENV["PYTHON"] = ""
Pkg.build("PyCall")
using Conda
Conda.add("numpy"; channel="conda-forge")
Conda.add("matplotlib")

# Mock Gurobi dependency
try
  Pkg.add("Gurobi")
  using Gurobi
catch e
  println("Gurobi not available, proceeding with mock.")
end
#Pkg.activate(".")

#Pkg.add(url="https://github.com/hradet/Metaheuristics.jl")

include("../src/Genesys2.jl")
using Documenter, Main.Genesys

makedocs(sitename="Microgrid_Genesys.jl", format = Documenter.HTML(prettyurls = false))

deploydocs(
  repo = "github.com/Cboennec/Microgrid_Genesys.jl.git",
  branch = "gh-pages",
  target = "build",
  #key = ENV["DOCUMENTER_KEY"]
)