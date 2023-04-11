using Pkg

ENV["PYTHON"] = ""
Pkg.build("PyCall")
using Conda
Conda.add("numpy"; channel="conda-forge")

Pkg.activate(".")

Pkg.add(url="https://github.com/hradet/Metaheuristics.jl")

include("src/Genesys.jl")

using Documenter
using Main.Genesys

makedocs(sitename="Microgrid Genesys",
          modules=[Genesys],
          format=Documenter.HTML(),
          pages =  pages = Any[
            "Home" => "src/index.md",
            "Page 1" => "src/page1.md",
            "Assets" => "src/assets.md"
            ])



