include(joinpath(pwd(),"src","Genesys2.jl"))


nh, ny, ns = 8760, 10, 4

pygui(true)
plotlyjs()


microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion(),
                Grid(carrier = Electricity()))
                

using JLD2, FileIO

data_optim = JLD2.load(joinpath(pwd(),"data_light_4.jld2"))
            
ω_a = Scenarios(microgrid, data_optim; same_year=true, seed=1:ns)
            
generations = Dict("Solar" => 10.)
storages = Dict("Liion" => 40.)
subscribed_power = Dict("Electricity" => 10.)

designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω_a)

controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = 2)), ω_a)

simulate!(microgrid, controller, designer, ω_a, options = Options(mode = "serial"))


metrics = Metrics(microgrid, designer)

    
plot_operation2(microgrid, y=1:ny, s=1:1)

