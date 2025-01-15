using Pkg

# Activate and instantiate the docs environment
Pkg.activate(".")
Pkg.instantiate()

ENV["PYTHON"] = ""
Pkg.build("PyCall")
using Conda
Conda.add("numpy"; channel="conda-forge")
Conda.add("matplotlib")
 
# Load Documenter and your package
using Documenter
using Genesys

# Generate the documentation
makedocs(
    sitename = "Microgrid_Genesys.jl",
    format = Documenter.HTML(prettyurls = false)
)

# Deploy the documentation
deploydocs(
    repo = "github.com/Cboennec/Microgrid_Genesys.jl.git",
    branch = "gh-pages",
    target = "build",
)
# import Pkg
# using Pkg

# # Activez l'environnement des `docs`
# #Pkg.activate(".")
# #Pkg.develop(PackageSpec(path = ".."))
# #Pkg.instantiate()


# ENV["PYTHON"] = ""
# Pkg.build("PyCall")
# using Conda
# Conda.add("numpy"; channel="conda-forge")
# Conda.add("matplotlib")
 
# include("../src/Genesys.jl")
# using Documenter

# makedocs(sitename="Microgrid_Genesys.jl", format = Documenter.HTML(prettyurls = false))

# deploydocs(
#   repo = "github.com/Cboennec/Microgrid_Genesys.jl.git",
#   branch = "gh-pages",
#   target = "build",
#   #key = ENV["DOCUMENTER_KEY"]
# )