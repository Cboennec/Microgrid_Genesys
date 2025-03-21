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
    Δh::Float64 # operations time step in hours
    nh::Int64 # number of operation stages
    Δy::Int64 # investment time step in years
    ny::Int64 # number of investment stages
    τ::Float64 # discount rate
    renewable_share::Float64 # share of renewables [0,1]

    GlobalParameters(nh, ny, ns;
                Δh = 1.,
                Δy = 1,
                τ = 0.045,
                renewable_share = 0.) = renewable_share == 1. ? new(ns, Δh, nh, Δy, ny, τ, 0.9999) : new(ns, Δh, nh, Δy, ny, τ, renewable_share)
end


"""
    Microgrid()

Structure containing every element/component of the microgrid including immaterial ones like electrical
demands.

# Fields:
- `parameters::GlobalParameters`
- `demands::Vector{AbstractDemand}`
- `generations::Vector{AbstractGeneration}` 
- `storages::Vector{AbstractStorage}`
- `converters::Vector{AbstractConverter}`
- `grids::Vector{AbstractGrid}`

Elements are divided in 5 categories and stored in 5 vectors with one cell for each element of a type :
* demands inheriting from AbstractDemand and containing a demand one or multiple `Main.Genesys.EnergyCarrier` (Electricity, Heat ,Hydrogen);
* generations inheriting from AbstractGeneration and containing energy production/generation element;
* storages inheriting from AbstractStorage and containing element responsible for the storage of different energy vectors;
* converters inheriting from AbstractConverter and containing any element that transform an energy vector into another one;
* grids inheriting from AbstractGrid and containing elements that sell or buy energy vectors.
(usually an external grid from which can be bought and sold energy);

These assets (components) can be later added with the add!(mg::Microgrid, assets...) 
See also `Main.Genesys.add!` for a step by step declaration.
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
 resulting in 3D arrays for usage in contexte see [Example section](Example.md#Constructing the grid)
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
#function preallocate!(mg::Microgrid, designer::AbstractDesigner)
#    designer.decisions = (generations = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.generations],
#                          storages = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.storages],
#                          converters = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.converters],
#                          subscribed_power = [zeros(mg.parameters.ny, mg.parameters.ns) for a in mg.grids])
#end


"""
    preallocate!(mg::Microgrid, designer::AbstractDesignerNonFloat)

Preallocate abstract designer with not only float decisions

Data struct are sized according to the number of scenario and years
 to store for each element to size a decision per investment stage.
"""
function preallocate!(mg::Microgrid, designer::AbstractDesigner)
    generations_dict = Dict()
    storages_dict = Dict()
    converter_dict = Dict()
    subscribed_power_dict = Dict()

    for a in mg.generations
        generations_dict[string(typeof(a))] = zeros(mg.parameters.ny, mg.parameters.ns)
    end

    for a in mg.storages
        storages_dict[string(typeof(a))] = zeros(mg.parameters.ny, mg.parameters.ns)
    end

    #Different decision type for FuelCells and Electrolyzer
    if designer isa MILP
        for a in mg.converters
            converter_dict[string(typeof(a))] = zeros(mg.parameters.ny, mg.parameters.ns)
        end
    else
        for a in mg.converters
            if a isa FuelCell
                converter_dict[string(typeof(a))] = (surface = zeros(mg.parameters.ny, mg.parameters.ns), N_cell = zeros(mg.parameters.ny, mg.parameters.ns))
            elseif a isa Electrolyzer
                converter_dict[string(typeof(a))] = (surface = zeros(mg.parameters.ny, mg.parameters.ns), N_cell = zeros(mg.parameters.ny, mg.parameters.ns))
            elseif a isa Heater
                converter_dict[string(typeof(a))] = (power = zeros(mg.parameters.ny, mg.parameters.ns))
            end
        end
    end
    
    for a in mg.grids
        subscribed_power_dict[string(typeof(a.carrier))] = zeros(mg.parameters.ny, mg.parameters.ns)
    end

    designer.decisions = (generations = generations_dict,
                          storages = storages_dict,
                          converters = converter_dict,
                          subscribed_power = subscribed_power_dict)
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

# serie_a : the serie you know a value on
# serie_b : the serie you want to find a value on
# a : the value you have
# b : the value you want
# increasing : is serie a increasing ?
function interpolation(serie_a::Vector{Float64}, serie_b::Vector{Float64}, a::Float64, serie_a_increasing::Bool)
   

    if serie_a_increasing
        if a <= serie_a[1]
            return serie_b[1]
        elseif a >= serie_a[end]
            return serie_b[end]
        else
            id = findfirst(serie_a .>= a)
        end
    else
        if a >= serie_a[1]
            return serie_b[1]
        elseif a <= serie_a[end]
            return serie_b[end]
        else
            id = findfirst(serie_a .<= a)
        end
    end

    if id == 1
        if a == serie_a[1]
            return b[1]
        end
    end
   
    frac = (a - serie_a[id-1]) / (serie_a[id] - serie_a[id-1])
    b = frac * (serie_b[id] - serie_b[id-1]) + serie_b[id-1]
    return b
end


function get_id_dict(mg::Microgrid)

    id_dict = Dict()
    id_dict["storages"] = [typeof(a) for a in mg.storages]
    id_dict["generations"] = [typeof(a) for a in mg.generations]
    id_dict["converters"] = [typeof(a) for a in mg.converters]
    return id_dict 
end

