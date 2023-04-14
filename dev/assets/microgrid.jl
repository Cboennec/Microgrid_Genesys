"""
    mutable struct GlobalParameters

Structure containing global parameter for the simulation and optimization.

This struct is contained inside the Microgrid struct and provide global parameters for
sizing and simulating.
# Parameters :
* (in sizing context) the number of operation and investment stages (ny, nh) or 
(in simulation context) Time horizon and number of operation per year (ny, nh); 
* the number of scenario (ns);
* *optionnal* : the yearly and hourly time steps (Δh, Δy) [default = (1,1)];
* *optionnal* : the inlation rate (τ) [default = 0.045];
* *optionnal* : the minimum renewable_share [default = 0.]"""
mutable struct GlobalParameters
    ns::Int64 # number of [nh, ny] scenarios
    Δh::Int64 # operations time step in hours
    nh::Int64 # number of operation stages
    Δy::Int64 # investment time step in years
    ny::Int64 # number of investment stages
    τ::Float64 # discount rate
    renewable_share::Float64 # share of renewables [0,1]

    GlobalParameters(nh, ny, ns;
                Δh = 1,
                Δy = 1,
                τ = 0.045,
                renewable_share = 0.) = renewable_share == 1. ? new(ns, Δh, nh, Δy, ny, τ, 0.9999) : new(ns, Δh, nh, Δy, ny, τ, renewable_share)
end


"""
    Microgrid()

Structure containing every element of the grid including immaterial ones like electrical
 demands.

Those elements are divided in 5 types and stored in 5 vectors with one field for each element of a type :
* demands inheriting from AbstractDemand and containing a demand for each energy (Electricity and Heat);
* generations inheriting from AbstractGeneration and containing energy 
production/generation assets;
* storages inheriting from AbstractStorage and containing energy storage assets;
* converters inheriting from AbstractConverter and containing energy convertion assets;
* grids inheriting from AbstractGrid and containing energy market assets 
(usually an external grid from which can be bought and sold energy);

These assets can be later added with the add!(mg::Microgrid, assets...) 
See also [`Main.Genesys.add!`](@ref) for a step by step declaration
"""
mutable struct Microgrid
    parameters::GlobalParameters
    demands::Vector{AbstractDemand}
    generations::Vector{AbstractGeneration}
    storages::Vector{AbstractStorage}
    converters::Vector{AbstractConverter}
    grids::Vector{AbstractGrid}

    Microgrid(; parameters = GlobalParameters(8760, 20, 1)) = new(parameters)
end



"""
    add!(mg::Microgrid, assets...)


Add to the given microgrid *mg* every asset listed as parameter into the appropriate 
vector.

Meanwhile data struct are created,
they are sized according to the number of scenario, years, and hours per year, ns, ny, nh
 resulting in 3D arrays
"""
function add!(mg::Microgrid, assets...)
    mg.demands = [preallocate!(a, mg.parameters.nh, mg.parameters.ny, mg.parameters.ns) for a in assets if a isa AbstractDemand]
    mg.generations = [preallocate!(a, mg.parameters.nh, mg.parameters.ny, mg.parameters.ns) for a in assets if a isa AbstractGeneration]
    mg.storages = [preallocate!(a, mg.parameters.nh, mg.parameters.ny, mg.parameters.ns) for a in assets if a isa AbstractStorage]
    mg.converters = [preallocate!(a, mg.parameters.nh, mg.parameters.ny, mg.parameters.ns) for a in assets if a isa AbstractConverter]
    mg.grids = [preallocate!(a, mg.parameters.nh, mg.parameters.ny, mg.parameters.ns) for a in assets if a isa AbstractGrid]
end

"""
    preallocate!(mg::Microgrid, designer::AbstractDesigner)

Preallocate abstract designer

Data struct are sized according to the number of scenario and years
 to store for each element to size a decision per investment stage.
"""
function preallocate!(mg::Microgrid, designer::AbstractDesigner)
    designer.decisions = (generations = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.generations],
                          storages = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.storages],
                          converters = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.converters],
                          subscribed_power = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.grids])
end

"""
    preallocate!(mg::Microgrid, controller::AbstractController)


Preallocate abstract controller

Data struct are sized according to the number of scenario, years and hours per year 
to store for each element to size a decision per operation stage.
"""
function preallocate!(mg::Microgrid, controller::AbstractController)
    controller.decisions = (converters = [zeros(mg.parameters.nh, mg.parameters.ny, mg.parameters.ns) for a in mg.converters],
                            storages = [zeros(mg.parameters.nh, mg.parameters.ny, mg.parameters.ns) for a in mg.storages])
end

"""
    isin(field::Vector, type::DataType)

Find if the datatype is in a mg field and where it is

For a given vector *field* and a Datatype *type*, if an object of the type *type*
is in *field* return true and the index of the type in the vector else return false
and NaN.
"""
function isin(field::Vector, type::DataType)
    # Return true if the datatype is in the field and its index
    bool, idx = false, NaN
    for (k, a) in enumerate(field)
        if a isa type
            bool, idx = true, k
        end
    end
    return bool, idx
end
