
# Definition du type individu
mutable struct Individu
    front::Int                     # front d'appartenance
    niche_count::Float64           # indice de densite
    val_param::Vector{Float64}     # parametres
    contrainte::Vector{Float64}    # contraintes
    critere::Vector{Float64}       # criteres
    type_cross::Int                # X-gene : type de croisement
    realisable::Bool               # = 1 si realisable
end


# Definition du type type_param
mutable struct TypeParam
    pmin::Float64
    pmax::Float64
end

# Struct for tracking the evolution  of cross operators
mutable struct Cross
    cross_bgx::Vector{Int}
    cross_blx::Vector{Int}
    cross_sbx::Vector{Int}
end


mutable struct NSGAII <: AbstractMetaheuristic

    #Problem dependant
    lb_var::Vector{Float64} # Borne inferieur des variables
    ub_var::Vector{Float64} # Borne supérieur des variables

    nb_param::Int # Nombre de paramètre
    nb_gene::Int  # nombre de generation
    nb_ind::Int # Nombre d'individu dans la population
    nb_pop::Int # Taille de la population courrante + archive


    pm::Float64  # taux de mutation - defaut: 1.0/nb_param
    pm_cross::Float64 # taux de mutation du X-gene - defaut: 5%
    type_cross::Int # type de croisement - defaut : 4
                                            # 1 = BGX
                                            # 2 = SBX
                                            # 3 = BLX
                                            # autre = autoadaptatif 

    dec::Float64         # parametre du BLX - defaut: 0.5                    
    eta::Float64         # parametre du SBX - defaut: 1.0
    k::Int               # parametre du BGX - defaut: 16
    discrete::Bool      # parametre de mutation - defaut VRAI  

    function NSGAII(lb_var, ub_var;
            nb_gene = 100,
            nb_ind = 100,
            pm_cross = 0.05,
            type_cross = 4,
            dec = 0.5,
            eta = 1.0,
            k=16,
            discrete = true)

            @assert(length(lb_var) == length(ub_var))
            nb_param = length(lb_var)
            nb_pop = 2 * nb_ind
            pm = 1.0/nb_param

            new(lb_var, ub_var, nb_param, nb_gene, nb_ind, nb_pop, pm, pm_cross, type_cross, dec, eta, k, discrete)
    end 


end


mutable struct NSGAIISensitivityAnalysis
    SobolFirst::Matrix{Float64}
    Pearson::Matrix{Float64}
end



mutable struct NSGAIIResults <: MetaheuristicResults
    method::NSGAII
    pop_track::Vector{Individu}
    population::Vector{Individu}
    #sensitivity::NSGAIISensitivityAnalysis
end

