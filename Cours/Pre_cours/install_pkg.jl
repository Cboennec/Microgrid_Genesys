
import Pkg
using Pkg


#Set up python pre installations can take some time
ENV["PYTHON"] = ""
Pkg.add("PyCall")
Pkg.build("PyCall")

Pkg.add("Conda")
using Conda   

Conda.add("numpy"; channel="anaconda")
Conda.add("matplotlib")
Conda.add("Pandas")

using PyCall

ENV["PATH"] = Conda.bin_dir(Conda.ROOTENV) * ";" * ENV["PATH"]
Pkg.add("PyPlot")
using PyPlot

Conda.add("Seaborn")



#Opti pkg
Pkg.add("JuMP")
Pkg.add("Cbc")
Pkg.add("Ipopt")
Pkg.add("SDDP")
Pkg.add("Metaheuristics")
#Pkg.add(url="https://github.com/hradet/Metaheuristics.jl")
using JuMP, Cbc, Ipopt, SDDP, Metaheuristics

#Math pkg
Pkg.add("Statistics")
Pkg.add("StatsBase")
Pkg.add("MultivariateStats")
Pkg.add("Clustering")
Pkg.add("Distributions")
Pkg.add("Distances")
Pkg.add("LinearAlgebra")
Pkg.add("Interpolations")

using StatsBase,  Statistics, MultivariateStats, Clustering, Distributions, Distances, LinearAlgebra, Interpolations

#Other (file, data, plot, display progression ...)


Pkg.add("Seaborn")
Pkg.add("Pandas")
Pkg.add("ProgressMeter")
Pkg.add("Dates")
Pkg.add("Distributed")
Pkg.add("SharedArrays")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Random")
Pkg.add("JLD2")
Pkg.add("FileIO")
Pkg.add("Plots")
Pkg.add("PlotlyJS")


using Pandas, Seaborn, ProgressMeter, Dates, Distributed, SharedArrays, CSV, DataFrames, Random, JLD2, FileIO, Plots, PlotlyJS

Pkg.status()


