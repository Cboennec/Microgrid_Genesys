include(joinpath("..","..","src","Genesys2.jl"))



#Degradation curves μV/h as a function of the current density for different current densities
# See (https://oatao.univ-toulouse.fr/29665/1/Pessot_Alexandra.pdf) fig III.43
P_min = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_min.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
P_int = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_int.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
P_max = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","P_max.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))

#The voltage as a function of the current density at the beginning of life
V_J_FC_df = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","V_J_PAC.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
V_J_FC = zeros(2, length(V_J_FC_df.J)) #J, V, P
for (i,a) in enumerate([V_J_FC_df.J, V_J_FC_df.V])
    V_J_FC[i,:] = a 
end


V_J_Elyz_df = DataFrames.DataFrame(CSV.File(joinpath("Examples","data","V_J_Elyz.csv"), delim = ",", header = [Symbol("J"),Symbol("V")], types=Dict(:J=>Float64, :V=>Float64)))
V_J_Elyz = zeros(2, length(V_J_Elyz_df.J)) #J, V, P

for (i,a) in enumerate([V_J_Elyz_df.J, V_J_Elyz_df.V])
    V_J_Elyz[i,:] = a 
end

datas_deg_FC = [P_min,P_int,P_max]
current_densities = [0.075, 0.42, 0.62]

J_ref = 0.62
#The FuelCell for which we have data is pretty bad so we consider a fuel cell with 15000 hour lifetime for the reference current density.
obj_hours = 15000.

deg = create_deg_params(datas_deg_FC, current_densities, V_J_FC, J_ref, obj_hours)



data_optim = JLD2.load(joinpath("Cours", "Cours4", "Data_base_TP4.jld2"))

nh, ny, ns = 8760, 5, 2

plotlyjs() 

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

fc =  FuelCell(;V_J_ini = V_J_FC, SoH_model = PowerAgingFuelCell(;deg_params=deg, StartStop = true))
elyz = Electrolyzer(;V_J_ini = V_J_Elyz)
liion =  Liion(SoC_model = PolynomialLiionEfficiency(), SoH_model = SemiEmpiricalLiion())
# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                liion, H2Tank(),
                elyz, fc,
                Grid(carrier = Electricity()))
     
ω_opti = Scenarios(microgrid, data_optim; same_year=false, seed=hcat([x for x in 1:5], [x for x in 6:10]))

                    
 


    # Objectif function definition 
    
function get_obj(decisions, mg, designer, ω, varID)

    #println(decisions)


    nh, ny, ns = size(ω.demands[1].power)

    # Initialize mg
    mg_m = mg
    ω_m = ω

    mg_m.parameters = GlobalParameters(nh, ny, ns)

    # Initialize with the manual designer
    converters, generations, storages, subscribed_power = Dict(), Dict(), Dict(), Dict()
    for a in mg_m.converters
        if a isa Electrolyzer || a isa FuelCell
            converters[string(typeof(a))] = (surface = decisions[varID[string(typeof(a))].surface], N_cell=Int64(round(decisions[varID[string(typeof(a))].N_cell])))
        elseif a isa Heater
            converters[string(typeof(a))] = decisions[varID[string(typeof(a))]]
        end
    end
    for a in mg_m.generations
            generations[string(typeof(a))] = decisions[varID[string(typeof(a))]]
    end
    for a in mg_m.storages
            storages[string(typeof(a))] = decisions[varID[string(typeof(a))]]  
    end

    subscribed_power = Dict("Electricity" =>  decisions[varID["Electricity"]])

 
    designer_m = initialize_designer!(mg_m, Manual(generations = generations, storages = storages, converters = converters, subscribed_power = subscribed_power), ω_m)

    # Initialize controller
    controller_m = initialize_controller!(mg_m, RBC(options = RBCOptions(policy_selection = 7)), ω_m)

    # Simulate
    simulate!(mg_m, controller_m, designer_m, ω_m, options = Options(mode = "multithreads"))

    # Metrics
    metrics = Metrics(mg_m, designer_m)

 
    results = []



    push!(results, -mean(sum(metrics.npv.total, dims = 1))) #NPV
    push!(results, -mean(metrics.renewable_share)) #RES dist
    
    critere = [r for r in results]
    contraintes = [get_δ_eq(mg, Electricity), get_δ_eq(mg, Heat), get_δ_eq(mg, Hydrogen)]

    return contraintes, critere
    
end

varID = Dict("Solar" => 1,
"Liion" => 2,
"H2Tank" => 3,
"Electrolyzer" => (N_cell = 4, surface = 5), 
"FuelCell" => (N_cell = 6, surface = 7),
"Electricity" => 8)

ub_var = [80., 150., 2500., 5, 30., 5, 30., 36.]
lb_var = [0., .1,   0.,  0,  0.,  0, 0., 0.] 

NSGA_pop = 30 #100 recommended
NSGA_gen = 30 #100 recommended

# Les éléments importants :
# Le microréseau : microgrid
# Le scénario : ω_opti
# Les bornes pour les variables : ub_var, lb_var
# Le dictionnaire qui donne la corespondance entre le vecteur de décision et le design : varID
# La fonction objectif qui à pour but de retourner la valeur des objectif : f_obj (à noter que notre version de l'algorithme vise à minimiser adapter donc votre fonction obj en conséquence)
# La taille de la population et le nombre de génération : NSGA_pop, NSGA_gen
# Le controller utiliser pour operer le réseau : controller
    results, SA = generate_designers_MO(microgrid, 
    Metaheuristic(options = MetaheuristicOptions(;method = NSGAII(lb_var, ub_var; nb_ind = NSGA_pop), multithreads=true, iterations = NSGA_gen, controller = RBC(options = RBCOptions(policy_selection = 7)))),
    ω_opti,
    ub_var,
    lb_var,
    varID, f_obj = get_obj) 

    # Notez le changement de signe des objectif car notre problème est un problème de maximization (NPV et RES) et l'algorithme fonction avec des minimum 

    critere = -hcat([ind.critere for ind in results.results.population]...)
    param = hcat([ind.val_param for ind in results.results.population]...)
    crit_track = -hcat([ind.critere for ind in results.results.pop_track]...)
    param_track = hcat([ind.val_param for ind in results.results.pop_track]...)


    Plots.plot(critere[1,:], critere[2,:], seriestype=:scatter, label="front")

    
    order = sortperm(critere[1,:]) # NPV order
    critere_ordered = critere[:,order] # crit ordered with regard to the NPV 
    param_ordered = param[:,order] #  param ordered with regard to the NPV 


tmp = Dict("crit1" => critere_ordered[1,:], "crit2" => critere_ordered[2,:], "place" => [x for x in 1:length(critere_ordered[2,:])])
df = DataFrames.DataFrame(tmp)


PlotlyJS.plot(
        df, x=:crit1, y=:crit2, text=:place,
        mode="markers+text", size_max=60, kind="scatter",
        textposition="top center", labels=Dict(
            :crit1 => "NPV [€]",
            :crit2 => "RES",
            :place => "Rank"), PlotlyJS.Layout(
                height=800,
                title_text="Pareto Front",
                
    )
)





    


