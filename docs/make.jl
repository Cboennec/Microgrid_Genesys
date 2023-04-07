using Documenter
include("..\\src\\Genesys.jl")

using Main.Genesys

makedocs(
    root    = "C:\\Users\\coren\\OneDrive\\Bureau\\LAPLACE\\ThesisCB\\Codes\\Microgrid",
    source  = "docs//src",
    build   = "build",
    clean   = true,
    doctest = true,
    modules = Module[Genesys],
    repo    = "",
    highlightsig = true,
    sitename = "Genesys Microgrid Tool Documentation",
    expandfirst = [],
    pages = [
        "Index" => "index.md",
    ],
    draft = false,
)

