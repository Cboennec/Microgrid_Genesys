# Ce script est une proposition pour l'ajout d'un compresseur, auxiliaire de l'éléctrolyseur au package.
# La solution proposée implémente une catégorie d'AbstractAuxiliary et d'AbstractCompresseur afin de permettre la variation des modèles de compresseur.
# Le compresseur est ajouter au modèle d'efficacité de l'éléctrolyseur et une prise d'information (entre H2tank et compresseur) est organisée dans la boucle interne de simulation avant la mise en oeuvre des décision (compute_operation_dynamics)
# Le moment venu, tout les éléments sont disponnible au calcul du partage de puissance entre éléctrolyseur et compresseur.

include(joinpath("..","..","src","Genesys2.jl"))
# A noter que j'ai déjà réalisé une mise à jour dans laquel j'ai changé H2Tank pour ajouter une pression max.

############# 1 : Création d'une structure Compresseur Et d'un superype AbstractAuxiliary
abstract type AbstractAuxiliary end

abstract type AbstractCompresseur <: AbstractAuxiliary end

# Structure compresseur dont la part d'energy dépend de la pression
mutable struct Compresseur_P <: AbstractCompresseur 
    
    Cp_H2::Float64 #[J/kg/K]
    T_amb::Float64 # [K]
    η_cmp_H2::Float64 #
    P_ELYZ::Float64 #[barA ou Pa]
    γ::Float64 #
    P_max::Float64

    pression_H2::AbstractArray{Float64,1} # La pression actuelle dans le H2Tank

    Compresseur_P(;
    Cp_H2 = 14266.,
    T_amb = 293.,
    η_cmp_H2 = 0.9,
    P_ELYZ = 1., # ici en bar
    γ = 0.5,
    P_max = 100.
    ) = new(Cp_H2, T_amb, η_cmp_H2, P_ELYZ, γ, P_max)

end



# Structure de compresseur dont la part d'energie dépend de rien (la part est fixe)
mutable struct Compresseur_Fix <: AbstractCompresseur 
    
    η::Float64
    pression_H2::AbstractArray{Float64,1} # La pression actuelle dans le H2Tank

    Compresseur_Fix(;
    η = 0.1
    ) = new(η)

end

# 2 : Redéfinition des structure de model d'efficacité pour l'Electrolyzer, ajout d'un compresseur
mutable struct FixedElectrolyzerEfficiency2 <: AbstractElectrolyzerEffModel
    
    compresseur::AbstractCompresseur
    α_p::Float64 #Minimum power defined as a share of the maximum Power
    η_E_H2::Float64 #The efficiency from Electricity to DiHydrogen 
    η_E_H::Float64 #The efficiency from Electricity to Heat
    k_aux::Float64
    powerMax_ini::Float64
    couplage::Bool 
  
    powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the FuelCell
  
    V_J::AbstractArray{Float64,3}
  
    FixedElectrolyzerEfficiency2(;compresseur = Compresseur_Fix(),
    α_p = 0.05,
      η_E_H2 = 0.7,
      η_E_H = 0.,
      k_aux = 0.15,
      powerMax_ini =.00001,
      couplage = false
      ) = new(compresseur, α_p, η_E_H2, η_E_H, k_aux, powerMax_ini, couplage)
end
  
# Le 2 est simplement présent car je ne peux pas déclarer 2 structures du même nom et une structure PolarizationElectrolyzerEfficiency existe déjà au sein du package
mutable struct PolarizationElectrolyzerEfficiency2 <: AbstractElectrolyzerEffModel

    compresseur::AbstractCompresseur

    α_p::Float64 #Minimum power defined as a share of the maximum Power
    k_aux::Float64 # Share of the power used by the auxiliaries
    couplage::Bool
    K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant
    LHV::Float64  
    powerMax_ini::Float64

    powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the Electrolyzer
  
    V_J::AbstractArray{Float64,3}
  
    PolarizationElectrolyzerEfficiency2(;compresseur = Compresseur_P(),
    α_p = 0.05,
    k_aux = 0.15,
    couplage = true,
    K = (33.33 *  2.016 * 3600)  / (2*96485.3321),  #LHV * M_H2 * λ * 3600/(2*F)
    LHV = 33.33,
    powerMax_ini = .00001,
    ) = new(compresseur, α_p, k_aux, couplage, K, LHV, powerMax_ini)
end
  
# Le 2 est simplement présent car je ne peux pas déclarer 2 structures du même nom et une structure LinearElectrolyzerEfficiency existe déjà au sein du package
mutable struct LinearElectrolyzerEfficiency2 <: AbstractElectrolyzerEffModel
    compresseur::AbstractCompresseur
    α_p::Float64 #Minimum power defined as a share of the maximum Power
    k_aux::Float64 # Share of the power used by the auxiliaries
    couplage::Bool
    K::Float64 # Defined as a constant Latent Heat Value * masse molaire * stoechiometric coefficient / 2Faraday constant  
    powerMax_ini::Float64
  
    powerMax::AbstractArray{Float64,3} #The maximum power that can be demanded to the Electrolyzer
  
    a_η::Vector{Float64} # the slope for the fucntion η(P)
    b_η::Vector{Float64} # the ordinate at the origin for the function η(P)
  
    V_J::AbstractArray{Float64,3} # Polarization curve, One by scénario
  
  
    LinearElectrolyzerEfficiency2(;compresseur = Compresseur_Fix(),
     α_p = 0.05,
     k_aux = 0.15,
    couplage = true,
    K = (33.33 *  2.016 * 3600)  / (2*96485.3321),  #PCI * M_H2 * 3600/(2*F)
    powerMax_ini = .00001,
    ) = new(compresseur, α_p, k_aux, couplage, K, powerMax_ini)
end
  


# 3 : initialisation du tableau qui receuille la valeur de pression actuelle du tank (1 valeure par scénario)

### Preallocation
function preallocate!(elyz::Electrolyzer, nh::Int64, ny::Int64, ns::Int64)
    elyz.EffModel.powerMax = convert(SharedArray,zeros(nh+1, ny+1, ns)) ;  elyz.EffModel.powerMax[1,1,:] .= elyz.EffModel.powerMax_ini
    elyz.η = convert(SharedArray,zeros(nh+1, ny+1, ns))
  
    elyz.carrier = [Electricity(), Heat(), Hydrogen()]
    elyz.carrier[1].power = convert(SharedArray,zeros(nh, ny, ns))
    elyz.carrier[2].power = convert(SharedArray,zeros(nh, ny, ns))
    elyz.carrier[3].power = convert(SharedArray,zeros(nh, ny, ns))
    elyz.soh = convert(SharedArray,zeros(nh+1, ny+1, ns)) ; elyz.soh[1,1,:] .= elyz.soh_ini
    elyz.cost = convert(SharedArray,zeros(ny, ns))
  
    elyz.SoH_model.V_J_ini = convert(SharedArray, zeros(3, length(elyz.V_J_ini[1,:]))) #J, V, P
    elyz.SoH_model.V_J = convert(SharedArray, zeros(3, length(elyz.V_J_ini[1,:]), ns)) #J, V, P
    elyz.EffModel.V_J = convert(SharedArray, zeros(3, length(elyz.V_J_ini[1,:]), ns)) #J, V, P
  
    if elyz.EffModel isa LinearElectrolyzerEfficiency
        elyz.EffModel.a_η =  convert(SharedArray, zeros(ns))
        elyz.EffModel.b_η = convert(SharedArray, zeros(ns))
    end

    elyz.EffModel.compresseur.pression_H2 = convert(SharedArray, zeros(ns))

    return elyz
end


# 4 : Ajout de la communication entre Elyz et H2tank dans la boucle interne de simulation. 
# On ajoute ce partage d'inforamtion entre update_operation_informations et compute_operation_decisions
# Après la récupération d'info du scénario et avant le calcul des décisions par la RB.
function simulate!(h::Int64,
        y::Int64,
        s::Int64,
        mg::Microgrid,
        controller::AbstractController,
        designer::AbstractDesigner,
        ω_simu::AbstractScenarios,
        options::Options)

    # Update operation informations
    update_operation_informations!(h, y, s, mg, ω_simu)

    # Share informations between assets (ex : H2tank and Electrolyzer for splitting the power with the compressor)
    share_info_bw_assets!(h, y, s, mg)

    # Compute operation decision variables
    compute_operation_decisions!(h, y, s, mg, controller)

    # Compute operation dynamics for each converter and storage in mg
    compute_operation_dynamics!(h, y, s, mg, controller)

    # Power balance constraint checked for each node (recourse variables)
    compute_power_balances!(h, y, s, mg)

end

# 5 : définition de la fonction de partage d'informations
# Mis à jour dans la boucle interne de simu
function share_info_bw_assets!(h::Int64, y::Int64, s::Int64, mg::Microgrid)

    id_elyz = findfirst(a isa Electrolyzer for a in mg.converters)
    id_tank = findfirst(a isa H2Tank for a in mg.storages)
    if !isnothing(id_elyz) && !isnothing(id_tank)
        mg.converters[id_elyz].EffModel.compresseur.pression_H2[s] = get_pression(mg.storages[id_tank], h,y,s)
    end
end

# 6 : définition de la fonction de pression du tank.
# Une fonction d'exemple du calcul de pression (ici la pression est une fonction lineaire de l'état de charge avec 1 bar minimum)
function get_pression(tank::H2Tank, h::Int64, y::Int64, s::Int64)
    return min(1 + tank.soc[h,y,s] * tank.pression_max, tank.pression_max)
end

# 7 : définition de la fonction de calcul de la part d'energie allouée au compresseur pour chaque type de compresseur
# Return k_aux for a compressor of type Compresseur_Fix
function get_compressor_share(power_E::Float64, compresseur::Compresseur_Fix, s::Int64)
    return compresseur.η
end

# Return k_aux for a compressor of type Compresseur_P
function get_compressor_share(power_E::Float64, compresseur::Compresseur_P, s::Int64)

    share = max(log(compresseur.pression_H2[s]) * 3 + 2, 2.)

    return share
end

# 8 : définition de la fonction de calcul du rendement de l'electrolyzer en prenant en compte la part du compresseur.
# Redéfinition des fonctions pour prendre en compte le compresseur
function compute_operation_efficiency(elyz::Electrolyzer, model::PolarizationElectrolyzerEfficiency2, h::Int64,  y::Int64,  s::Int64, decision::Float64)

    #Apply minimum power
    model.powerMax[h,y,s] * elyz.min_part_load <= -decision ? power_E = max(decision, -model.powerMax[h,y,s]) : power_E = 0. 
    if power_E < 0
      #Compute the remaining power after feeding the auxiliaries 

        k_aux = get_compressor_share(power_E, model.compresseur, s)

        P_elyz = ceil(power_E / (1 + k_aux); digits=6)

        j = interpolation(model.V_J[3,:,s], model.V_J[1,:,s], -P_elyz, true)

        i = j * elyz.surface

        #Power_out/power_in (per cell)
        η_E_H2 = i * model.K / (-power_E/ elyz.N_cell)     

        elyz.η[h,y,s] = η_E_H2

        η_E_H = 0.

        elec_power, heat_power, hydrogen_power = (power_E), - power_E * η_E_H,  - power_E * η_E_H2
  
    else 
      elec_power, heat_power, hydrogen_power = 0., 0., 0.
    end
   
    return elec_power, heat_power, hydrogen_power 
end




####################################################################################
####################################################################################
############################### test des models ####################################
####################################################################################
####################################################################################

nh,ny,ns = 8760, 10, 2


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

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

elyz = Electrolyzer(;V_J_ini = V_J_Elyz, EffModel = PolarizationElectrolyzerEfficiency2())
fc = FuelCell(;V_J_ini = V_J_FC, SoH_model = PowerAgingFuelCell(;deg_params=deg))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion(), H2Tank(α_soc_min=0.1),
                elyz, fc,
                Grid(carrier = Electricity()))
        
data_optim = JLD2.load(joinpath("Examples","data","ausgrid_1000_optim_1y.jld2"))

ω_a = Scenarios(microgrid, data_optim; same_year=false, seed=reshape(1:(ns*ny), (ny,ns)))


 conv = Dict("Electrolyzer" => (surface = 20., N_cell=1), "FuelCell" => (surface = 10. , N_cell=1))
 gen = Dict("Solar" => 60.)
 sto = Dict("Liion" => 60., "H2Tank" => 2500.)
 sub = Dict("Electricity" => 16.)

designer = initialize_designer!(microgrid, Manual(generations = gen, storages = sto, converters = conv, subscribed_power = sub), ω_a)

controller_rb = RBC(options = RBCOptions(policy_selection = 7))
controller_rb = initialize_controller!(microgrid, controller_rb, ω_a)

simulate!(microgrid, controller_rb, designer, ω_a, options = Options(mode = "serial"))

plotlyjs()
plot_operation2(microgrid, y=1:ny, s=1:1)

# pygui(true)
# plot_operation(microgrid, y=1:ny, s=1:ns)