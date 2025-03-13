##### Script de création de la banque de données #############
##### Peut contenir autant de champ que désiré ###########

# Cahier des charge : 

# 1 récupération des données 
# A partir des fichier CSV créer des dataframes avec les champs de temps et de pluie (RR) et d'irradiance et production (P et G(i))
# Pour les données de pluie, conserver aussi le champ qui donne la station météo.

# 2 restructuration des données
# Conserver uniquement la station "TOULOUSE-BLAGNAC"
# Remplacer les valeurs manquantes des séries par 0 avec :  coalesce.(data.RR, 0)
# Transormer les données sur le format horaire et  Créez un champ DateTime qui donne le timecode de chaque entrée
# Pour les puissance transformez en kW par kWc, les données initiales sont en W 
# Conserver uniquement les année communes entre les deux dataframes
# Retirer les 29 fevrier pour avoir des années de 8760 heures


# 3 création de la banque de données
# Récuperer des données depuis notre vrai banque de donnée pour avoir des profil de conso d'Elec et de chaleur
# Reshape les données pour avoir (nh*1*ns)
# Création de la banque de données sous forme de dictionnaire
# Sauvegarde du dictionnaire au format JLD2.




using FileIO, CSV, DataFrames, Dates, JLD2


#################### Donnée de pluie ####################################################
############ https://meteo.data.gouv.fr/datasets/6569ad61106d1679c93cdf77 ################
#######################Fichier :  "MN_31_2010-2019.csv" #####################################
########################################################################################
# !!!!! # le fichier donne les volume de pluie toute les 6 minutes.


# On charge le fichier et on garde les colomne 2, 3, 4 respectivement le nom de la station météo le timecode, le volume de pluie en milimètre
file = CSV.File(joinpath(pwd(), "Cours", "Cours4", "pluie.csv"); select=[2, 3, 4])
# On en fait une dataframe
data = DataFrames.DataFrame(file)
# On filtre pour ne garder que les données de la station la plus proche
filter!(:NOM_USUEL =>  name -> name == "TOULOUSE-BLAGNAC", data) # name -> name == "TOULOUSE-BLAGNAC" est une fonction anonyme
# On remplace les valeures manquantes par 0
data.RR = coalesce.(data.RR, 0)
# On retire les minutes du timecode en ne gardant que les 10 premier charactère de AAAAMMJJHHMN
data.AAAAMMJJHHMN = SubString.(string.(data.AAAAMMJJHHMN),1,10)
# On transforme le timecode en un format de date plus lisible (On rajoute par la même une colonne)
data.DateTime = DateTime.(string.(data.AAAAMMJJHHMN), "yyyymmddHH")

# On groupe les ligne en faisant la somme de celle qui on le même datetime (donc par heure)
hourly_df_pluie = combine(DataFrames.groupby(data, :DateTime),  :RR => sum)

#####################################################################################
############## Chargement de donnée de production PV et d'ensoleillement ############
############## source   https://re.jrc.ec.europa.eu/pvg_tools/en/ ###################
############## param : 
# Lat, Lon : 43.603, 1.464
# Solar radiation database : PVGIS-SARAH2
# year:  2005 - 2020
# Mounting type: Fixed
# Slope [°] : Optimize slope (37)
# Azimuth [°] : Optimize Azimuth (1)
# PV power : On
# PV technology : Crystalline silicon
# Installed peak PV power [kWp] : 10
# System loss [%] : 14
######### Fichier : TLSE_PV_prod_Irradiation_2005_2020.csv ##########################

# On charge le fichier en selectionnant les 3 première colonne respectivement Datetime, PV production par kWc installé [W], irradiance[W/m^2]
file = CSV.File(joinpath(pwd(),  "Cours", "Cours4", "TLSE_prod_PV_2005_2020.csv"); delim=";", header=17, skipto=18, select=[1,2,3,5] )
# On en fait une DataFrame
data = DataFrames.DataFrame(file)
# On tronque les minutes
data.time = SubString.(string.(data.time),1,11)
# On transforme le format de date
data[!,:DateTime] = DateTime.(string.(data.time), "yyyymmdd:HH")
# On supprime l'ancienne colonne
select!(data, Not(:time))
# On passe les données de puissance en kW
data.P = data.P / 1000 / 10# 10 kWp normalisé

hourly_df_soleil = data

# On récupère les années min et max 
max_year = min(maximum(year.(hourly_df_soleil.DateTime)), maximum(year.(hourly_df_pluie.DateTime)))
min_year = max(minimum(year.(hourly_df_soleil.DateTime)), minimum(year.(hourly_df_pluie.DateTime)))
# On conserve les années communes
filter!(:DateTime =>  date -> max_year >= year(date) >= min_year , hourly_df_soleil)
filter!(:DateTime =>  date -> max_year >= year(date) >= min_year , hourly_df_pluie)
#On oublie les 29 fevrier
filter!(:DateTime =>  date -> !(day(date) == 29 && month(date) == 2), hourly_df_soleil)
filter!(:DateTime =>  date -> !(day(date) == 29 && month(date) == 2), hourly_df_pluie)











# Parameters (seul le nombre de scénario est supposé changer) le nombre d'année c'est 2 mais on en utilisera que 1.
const nh, ny, ns = 8760, 1, 10

# Ces données vont venir completer ce que l'on vient de collecter (elles ne sont pas en cohérence avec le reste car elles ne proviennent pas de Toulouse)
data2 = JLD2.load(joinpath(pwd(),"Cours","Cours4","ausgrid_10_optim_cours4.jld2"))

timestamp = repeat(reshape(hourly_df_pluie.DateTime, (nh,1,ns)), inner=(1,1,1))

ld_E = data2["ld_E"]["power"][:,:,1:ns]
ld_H = data2["ld_H"]["power"][:,:,1:ns]

# Mise en forme des données (on a besoin de 2 ans alors qu'on en utilise 1) 
pv_p = reshape(hourly_df_soleil.P, (nh,1,ns))
barrage_irr = reshape(hourly_df_soleil[!,"G(i)"], (nh,1,ns))
barrage_rain = reshape(hourly_df_pluie.RR_sum, (nh,1,ns))

# Scenarios
ω_barrage = Dict(
"dam" => Dict("t" => timestamp, "irradiance" => barrage_irr, "rain" => barrage_rain,  "cost" => 100 * ones(ny, ns)),
"pv" => Dict("t" => timestamp, "power" => pv_p, "cost" => 1300 * ones(ny, ns)),
"ld_E" => Dict("t" => timestamp, "power" => ld_E),
"ld_H" => Dict("t" => timestamp, "power" => ld_H),
"liion" => Dict("cost" => 300 * ones(ny, ns),),
"tes" => Dict("cost" => 10 * ones(ny, ns),),
"h2tank" => Dict("cost" => 10 * ones(ny, ns),),
"elyz" => Dict("cost" => 1300 * ones(ny, ns),),
"fc" => Dict("cost" => 1700 * ones(ny, ns),),
"heater" => Dict("cost" => 10 * ones(ny, ns),),
"grid_Elec" => Dict("t" => timestamp, "cost_in" => 0.19 * ones(nh, ny, ns), "cost_out" => 0.0001 * ones(nh, ny, ns), "cost_exceed" => 10.2 * ones(ny, ns))
)

JLD2.save(joinpath("Cours", "Cours4", "Data_base_TP4.jld2"), ω_barrage)



























########################################################################################
############# Proposition de Correction Banque de donnée ################################
#########################################################################################

file = CSV.File(joinpath(pwd(),  "Cours", "Cours4", "vent.csv"))
# On en fait une DataFrame
data = DataFrames.DataFrame(file)
# On garde l'Occitanie
filter!(:ville =>  name -> name == "Toulouse", data)
data.vent = data.var"wind m/s"

# On dedouble les champs pour avoir 1 entrée par heure
data = repeat(data, inner = 2)
data.DateTime[2:2:end] .+= Hour(1)

hourly_df_wind = data


nh, ny, ns = 8760, 1, 10

# Load input data
data2 = JLD2.load(joinpath(pwd(),"Cours","Cours4","ausgrid_10_optim_cours4.jld2"))

timestamp = repeat(reshape(hourly_df_pluie.DateTime, (nh,1,ns)), inner=(1,1,1))

ld_E = data2["ld_E"]["power"][:,1:1,1:ns]
ld_H = data2["ld_H"]["power"][:,1:1,1:ns]

# Mise en forme des données (on a besoin de 2 ans alors qu'on en utilise 1) 
pv_p = repeat(reshape(hourly_df_soleil.P, (nh,1,ns)), inner=(1,1,1))
barrage_irr = repeat(reshape(hourly_df_soleil.var"G(i)", (nh,1,ns)), inner=(1,1,1))
barrage_rain = repeat(reshape(hourly_df_pluie.RR_sum, (nh,1,ns)), inner=(1,1,1))
barrage_vent = repeat(reshape(hourly_df_wind.vent, (nh,1,ns)), inner=(1,1,1))
barrage_t2m = repeat(reshape(hourly_df_soleil.T2m, (nh,1,ns)), inner=(1,1,1))

# Scenarios
ω_barrage = Dict(
"dam" => Dict("t" => timestamp, "irradiance" => barrage_irr, "temperature" => barrage_t2m, "rain" => barrage_rain, "wind" => barrage_vent , "cost" => 100 * ones(ny, ns)),
"pv" => Dict("t" => timestamp, "power" => pv_p, "cost" => 1300 * ones(ny, ns)),
"ld_E" => Dict("t" => timestamp, "power" => ld_E),
"ld_H" => Dict("t" => timestamp, "power" => ld_H),
"liion" => Dict("cost" => 300 * ones(ny, ns),),
"tes" => Dict("cost" => 10 * ones(ny, ns),),
"h2tank" => Dict("cost" => 10 * ones(ny, ns),),
"elyz" => Dict("cost" => 1300 * ones(ny, ns),),
"fc" => Dict("cost" => 1700 * ones(ny, ns),),
"heater" => Dict("cost" => 0 * ones(ny, ns),),
"grid_Elec" => Dict("t" => timestamp, "cost_in" => 0.19 * ones(nh, ny, ns), "cost_out" => 0.0001 * ones(nh, ny, ns), "cost_exceed" => 10.2 * ones(ny, ns))
)

JLD2.save(joinpath("Cours", "Cours4", "Data_base_TP4.jld2"), ω_barrage)

