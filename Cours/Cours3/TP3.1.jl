
include(joinpath(pwd(),"src","Genesys2.jl"))



pyplot_installed = true
if !(isdir(Pkg.dir("PyPlot")))
    Pkg.add("plotlyjs")
    using plotlyjs
    plotlyjs()
    pyplot_installed = false
else
    using PyPlot
    pygui(true)
end


nh, ny, ns = 8760, 10, 1

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion( SoH_model = ModelTP3SOH(), eff_model = ModelTP3SOC()), 
                Grid(carrier = Electricity()))



using JLD2, FileIO

data_optim = JLD2.load(joinpath(pwd(), "Cours", "Cours3", "data_light_4.jld2"))
        

# Load data Scenarios
ω_a = Scenarios(microgrid, data_optim, true)
            
generations = Dict("Solar" => 25.)
storages = Dict("Liion" => 40.)
subscribed_power = Dict("Electricity" => 10.)
                


designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω_a)

controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = 2)), ω_a)

simulate!(microgrid, controller, designer, ω_a, options = Options(mode = "serial"))


metrics = Metrics(microgrid, designer)
    
plot_operation(microgrid, y=1:ny, s=1:1)



 
# Faite un rendement par palier qui se comporte différement en fonction du C-rate.
# Réaliser 3 paliers régies par des équations différente (vous pouvez ou non différencier la charge de la décharge)
# La seule cond  ition :  η ∈ [0-1]
# Attention, par défault les limites de SoC étant fixés à α_soc_max = 0.8 et α_soc_min = 0.2. et le pas de temps étant l'heure.
# La conséquence est que les bornes du C_rate sont implicitement définies comme ∈ [0-0.6]
mutable struct ModelTP3SOC <: AbstractLiionEffModel


end


# Creez un model dont le vieillissement vient du fait de rester en model charge ou en mode décharge.
# Une seule heure n'implique pas ou très peu de vieillissement puis on arrive à un vieillissement important pout plusieurs heures consécutives

mutable struct ModelTP3SOH <: AbstractLiionAgingModel

 
end



function get_nb_consecutif(soc::SharedArray{Float64, 3}, h::Int64, y::Int64, s::Int64)

    index = h
    tot = 0
    same_sign = true
    sign = (soc[index+1, y, s] - soc[index, y, s]) > 0 ? 1 : 0 # 0 negatif, 1 positif

    while index >= 1 && same_sign
        soc_diff = soc[index+1, y, s] - soc[index, y, s] 

        if soc_diff > 0
            new_sign = 1
        else 
            new_sign = 0
        end

        if new_sign != sign || soc_diff == 0
            same_sign = false
        else
            tot += 1
        end

        index -= 1 
    end

    return tot

end







###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
############### Correction model de batterie ##############################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################




# Faite un rendement par palier qui se comporte différement en fonction du C-rate.
# Réaliser 3 paliers régies par des équations différente (vous pouvez ou non différencier la charge de la décharge)
# La seule condition :  η ∈ [0-1]
# Attention, par défault les limites de SoC étant fixés à α_soc_max = 0.8 et α_soc_min = 0.2. et le pas de temps étant l'heure.
# La conséquence est que les bornes du C_rate sont implicitement définies comme ∈ [0-0.6]
mutable struct ModelTP3SOC <: AbstractLiionEffModel

	couplage::NamedTuple{(:E, :R), Tuple{Bool, Bool}}  #a boolean tuple to tell wether or not the soh should influence the other parameters.
	palier1::Float64 # En dessous on est en zone 1, au dessus en zone 2 ou 3
    palier2::Float64 # En dessous on est en zone 1 ou 2 au dessus on est en zone 3.

	
	ModelTP3SOC(;
    couplage = (E = true, R = true),
    palier1 = 0.3,
    palier2 = 0.5
		) = new(couplage, palier1, palier2)

end


# Doit retourner SoC, P_ch, P_dch
function compute_operation_soc(liion::Liion, model::ModelTP3SOC, h::Int64,  y::Int64,  s::Int64, decision::Float64, Δh::Float64)

    C_rate = abs(decision)/liion.Erated[y,s]


    if C_rate <= model.palier1 # schéma 1
        η = 1
    elseif C_rate > model.palier2 # schéma 3
        η = 0.8
    else # schéma 2
        η = 0.9
    end
    
	
	power_dch = max(min(decision, liion.soh[h,y,s] * liion.Erated[y,s] / Δh, η * (liion.soc[h,y,s] - liion.α_soc_min) * liion.Erated[y,s] / Δh), 0.)
	power_ch = min(max(decision, -liion.soh[h,y,s] * liion.Erated[y,s] / Δh, (liion.soc[h,y,s] - liion.α_soc_max) * liion.Erated[y,s] / Δh / η), 0.)

	return liion.soc[h,y,s] - (power_ch * η + power_dch / η) * Δh / liion.Erated[y,s], power_ch, power_dch

end














# Creez un model dont le vieillissement vient du fait de rester en model charge ou en mode décharge.
# Une seule heure n'implique pas ou très peu de vieillissement puis on arrive à un vieillissement important pout plusieurs heures consécutives

mutable struct ModelTP3SOH <: AbstractLiionAgingModel

    divider::Int64 # Le diviseur du nombre d'heure consécutives qui nous donne la dégradation. 
    # Une autre approche est bien sur possible. 

	ModelTP3SOH(;divider = 300000) = new(divider)
end


# La fonction va surtout avoir pour rôle de calculer le nombre d'heure succéssive de charge ou décharge. (Cela peut être réalisé dans un fonction annexe)
# Puis ce nombre va être traité pour induire une dégradation.

function compute_operation_soh(liion::Liion, model::ModelTP3SOH, h::Int64,y::Int64 ,s::Int64 , Δh::Float64)

    nb_consecutif = get_nb_consecutif(liion.soc, h, y, s)
	
    ΔSoH = nb_consecutif/model.divider

    return liion.soh[h,y,s] - ΔSoH

end







#################### test ############################


nh, ny, ns = 8760, 10, 1

microgrid = Microgrid(parameters = GlobalParameters(nh, ny, ns, renewable_share = .5))

# Add the equipment to the microgrid
add!(microgrid, Demand(carrier = Electricity()),
                Solar(),
                Liion( SoH_model = ModelTP3SOH(), eff_model = ModelTP3SOC()), 
                Grid(carrier = Electricity()))



using JLD2, FileIO

data_optim = JLD2.load(joinpath(pwd(), "Cours", "Cours3", "data_light_4.jld2"))
        

# Load data Scenarios
ω_a = Scenarios(microgrid, data_optim, true)
            
generations = Dict("Solar" => 25.)
storages = Dict("Liion" => 40.)
subscribed_power = Dict("Electricity" => 10.)
                


designer = initialize_designer!(microgrid, Manual(generations = generations, storages = storages, subscribed_power = subscribed_power), ω_a)

controller = initialize_controller!(microgrid, RBC(options = RBCOptions(policy_selection = 2)), ω_a)

simulate!(microgrid, controller, designer, ω_a, options = Options(mode = "serial"))


metrics = Metrics(microgrid, designer)
    
plot_operation(microgrid, y=1:ny, s=1:1)
