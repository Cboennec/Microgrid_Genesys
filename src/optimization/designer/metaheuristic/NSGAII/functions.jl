# Code NSGA-II avec Croisement autoadaptatif - Julia
# Bruno Sareni, LAPLACE - Mise a jour 2024 par Corentin Boennec
# References :  
# Deb et al, A fast Elitist Multiobjective Genetic Algorithm : NSGA-II
# IEEE Trans. on Evolutionary Computation, Vol 6, N�2, 2002
# Sareni et al, Recombination and Self-Adaptation in Multiobjective 
# Genetic Algorithms, LNCS, Vol 2936, 2004




# Definition du type individu
mutable struct Individu
    front::Int                     # front d'appartenance
    niche_count::Float64           # indice de densite
    val_param::Vector{Float64}     # parametres
    contrainte::Vector{Float64}   # contraintes
    critere::Vector{Float64}      # criteres
    type_cross::Int               # X-gene : type de croisement
    realisable::Bool             # = 1 si realisable
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

# Fonctions pour la generation des nombres aleatoires

# Fonction Randomize
function Randomize()
    Random.seed!(Int64(round(time())))
end

# Fonction Brandom
function Brandom()
    return rand(Bool)
end

# Fonction Rrandom
function Rrandom(mini, maxi)
    return rand() * (maxi - mini) + mini
end



#  Calcul des contraintes et des criteres - initialisation des Individus */ 

#  -------------------- fonction Calcule_criteres ---------------------- */
 
#  Calcul des critères et des contraintes associes a un Individu
#  Les contraintes doivent etre definies en termes de minimisation 
#  Elles sont affectees a zero si elles sont verifiees 
#  Elles sont d'autant plus positives qu'elles sont violees
#  ind -> realisable doit etre affecte a false lorsque une contrainte est 
#  violee
# Fonction Calcule_criteres
function Calcule_criteres(val_param::Vector{Float64}, f_eval::Function)
    # initialisations
    realisable = true               # par defaut realisable = true

    contraintes, criteres = f_eval(val_param)
    # calcul des contraintes

    # Annulation des contraintes negatives c.a.d verifiees
    for (i,c) in enumerate(contraintes)
        if c < 0
            contraintes[i] = 0
        end
    end

    # Affectation de la faisabilite
    for c in contraintes
        if c > 0
            realisable = false
            break
        end
    end

    # Calcul des criteres en cas de faisabilite
   # if realisable
    #    ind.critere = criteres
    #end

    return criteres, contraintes, realisable

end



#  ------------------- fonction Initialise_Individu -------------------- */

#  Initialise de facon uniformement aleatoire les valeurs des parametres 
#  à l'interieur des contraintes de domaine definissant l'espace de 
#  recherche.
#  Initilise de facon uniformement aleatoire les valeurs du X-gene codant
#  le type de croisement a appliquer.


# Fonction Initialise_population
function Initialise_population!( param::Array{TypeParam}, nb_param::Int, nb_pop::Int, f_eval::Function)
   
    val_param_ini = [Rrandom(param[j].pmin, param[j].pmax) for j in 1:nb_param]
    crit, cont, real = Calcule_criteres(val_param_ini, f_eval)

    pop = [Individu(0, 0.0, zeros(Float64, nb_param), zeros(Float64, length(cont)), zeros(Float64, length(crit)), 0, false) for _ in 1:nb_pop]
    pop[1].critere = crit
    pop[1].contrainte = cont
    pop[1].realisable = real
    pop[1].type_cross = rand(1:3)
    pop[1].val_param = val_param_ini


    for i in 2:nb_pop
        for j in 1:nb_param
            pop[i].val_param[j] = Rrandom(param[j].pmin, param[j].pmax)
            pop[i].type_cross = rand(1:3)
        end
        pop[i].critere, pop[i].contrainte, pop[i].realisable = Calcule_criteres(pop[i].val_param, f_eval)
    end

    return pop
end

####### fonctions necessaires a la dertermination de la dominance ###########
#  -------------------- fonction Dominance_individu -------------------- *
#  Etablissement de la dominance entre deux individus
#  Un individu realisable domine un individu non realisable
#  Si les 2 individus sont non realisables, la dominance est etablie dans
#  l'espace des contraintes
#  Si les 2 individus sont realisables, la dominance est etablie dans 
#  l'espace des critere

# Fonction Dominance_individu
function Dominance_individu(ind1::Individu, ind2::Individu)
    if ind1.realisable && !ind2.realisable # ind1 realisable / ind2 non realisable
        return true
    elseif !ind1.realisable && ind2.realisable # ind1 non realisable / ind2 realisable
        return false
    elseif !ind1.realisable && !ind2.realisable # ind1 et ind2 non realisables
        for i in 1:length(ind1.contrainte)
            if ind1.contrainte[i] > ind2.contrainte[i]
                return false
            end
        end
        return true
    else  # ind1 et ind2 realisables
        for i in 1:length(ind1.critere)
            # Si individu 1 à au moins un critère plus haut (moins bon en Min obj) alors il ne domine pas individu 2
            if ind1.critere[i] > ind2.critere[i]
                return false
            end
        end
        return true
    end
end



# Fonction Construction_MDSL (Matrice de Dominance au Sens Large)
function Construction_MDSL(pop::Vector{Individu}, nb_pop::Int)

    matrice_SL = ones(Int, nb_pop, nb_pop)

    for i in 1:nb_pop
        for j in 0:(nb_pop - 1 - i) 
            if !Dominance_individu(pop[i], pop[1+i+j])
                matrice_SL[i,i+j+1] = 0 
            end
        end
    end

    for i in 1:nb_pop
        for j in 0:(nb_pop-1-i)
            if !Dominance_individu(pop[1+i+j],pop[i])
                matrice_SL[i+j+1,i] = 0 
            end
        end
    end

    return matrice_SL
end


# Fonction Construction_MDSS (Matrice de Dominance au Sens Stricte)
function Construction_MDSS(pop::Array{Individu}, nb_pop::Int)
   
    
    matrice_SS = zeros(Int, nb_pop, nb_pop)
    matrice_SL = Construction_MDSL(pop, nb_pop)
  
    for i in 1:nb_pop 
        for j in 1:nb_pop 
            # Pour dominer au sens stricte il faut dominer sans être dominé
            matrice_SS[i,j] = matrice_SL[i,j]*(1 - matrice_SL[j,i])
        end
    end

    return matrice_SS

end

#---------------- fonction Echange_ligne_colonne --------------------- */
# Fonction Echange_ligne_colonne
# Echanges deux lignes et permute les colonnes correspondantes

function Echange_ligne_colonne(i1::Int, i2::Int , nb_ind::Int, matrice_SS::Matrix{Int})

  # permute lignes

    for k in 1:nb_ind
        tmp = matrice_SS[i1,k]
        matrice_SS[i1,k] = matrice_SS[i2,k]
        matrice_SS[i2,k] = tmp
    end

  # permute colonnes

    for k in 1:nb_ind
        tmp = matrice_SS[k,i1]
        matrice_SS[k,i1] = matrice_SS[k,i2]
        matrice_SS[k,i2] = tmp
    end

    return matrice_SS
end







# --------------------- fonction Determine_fronts --------------------- */
# Determine les fronts associes a chaque individu
# front = 0 = individu non domine
# Les individus sont également trié entre les 2 pop (archive et courrante).
function Determine_fronts(pop::Vector{Individu}, nb_pop::Int, frt::Ref{Int}, taille_front::Vector{Base.RefValue{Int}}, matrice_SS::Matrix{Int})

    frt[] = 0       # front courant 
    dec = 1        # decalage colonne dans la matrice 

    while true
  
        dec_front = dec
        taille_front[frt[] + 1][] = 0

        for j in dec_front:nb_pop
        
            #Faire cette partie
            id = findfirst(matrice_SS[dec_front:end,j] .== 1)
            i = (isnothing(id) ? id = nb_pop+1 : id)

            if i == (nb_pop+1)    # individu non domine sur ce front 
                taille_front[frt[] + 1][] += 1
                if j != dec # L'individu non dominé n'est pas le premier individu du front
                    matrice_SS = Echange_ligne_colonne(dec,j,nb_pop,matrice_SS) # On modifie la matrice et on inverse les individus
                    tmp = deepcopy(pop[dec])
                    pop[dec] = deepcopy(pop[j])
                    pop[j] = deepcopy(tmp)
                end
                pop[dec].front = frt[]
                dec += 1
            end
        end

        frt[] += 1
        if dec > nb_pop 
            break
        end
    end
end



# --------------------- fonction calcul_dominance --------------------- */

# affecte les front a tous les individus de la population
# Fonction Calcule_dominance
function Calcule_dominance!(pop::Array{Individu}, nb_pop::Int, nb_front::Ref{Int}, taille_front::Vector{Base.RefValue{Int}})

    # Qui domine qui, verifier la création de la matrice.
    matrice_DSS = Construction_MDSS(pop, nb_pop)

    Determine_fronts(pop, nb_pop, nb_front, taille_front, matrice_DSS)

end



# #  --------------------- fonction Affecte_niche ------------------------ *
# #  affecte a tous les individus un indice de densite par rapport a leur 
# #  position sur le front auquel ils appartiennent. Les individus les moins
# #  representes (les extremes) recoivent un indice infini. Plus densite au 
# #   voisinage d'un individu est forte, plus l'indice est petit.
  
function Affecte_niche(pop::Vector{Individu}, nb_pop::Int, nb_front::Base.RefValue{Int}, taille_front::Vector{Base.RefValue{Int}})
 
   

    dec = 1     # repere place sur le front courant 

    for j in 1:(nb_pop) # initialisation des niches
        pop[j].niche_count = 0
    end

    for i in 1:nb_front[]

        if pop[dec].realisable 
            # nichage en fonction des criteres
            if taille_front[i][] < 3
               # il y a un ou deux individus sur le front
                for j in 0:(taille_front[i][]-1)
                    pop[dec+j].niche_count = Inf
                end  
        
            else
                for index_glob in 1:length(pop[1].critere)
                    # tri des individus par ordre croissant des criteres  

                    pop[dec:(dec+taille_front[i][]-1)] = sort(pop[dec:(dec+taille_front[i][]-1)], by = x -> x.critere[index_glob])

                    diff_value = pop[dec+taille_front[i][]-1].critere[index_glob] - pop[dec].critere[index_glob]

                    # calcul du coefficient de surpeuplement seulement en cas de difference 
                    # entre les valeurs max et min du critere courant 

                    if diff_value > 0
                    
                        pop[dec].niche_count = Inf
                        pop[dec+taille_front[i][]-1].niche_count = Inf

                        for j in 1:(taille_front[i][]-1)
                            
                            if pop[dec+j].niche_count < Inf
                            
                                pop[dec+j].niche_count += (pop[dec+j+1].critere[index_glob] - pop[dec+j-1].critere[index_glob])/diff_value
                            end
                        end
                    end
                end
            end # fin else nb individus par front 
        # fin if individus realisable 
        else
           # nichage en fonction des contraintes 

            if taille_front[i][] < 3
                # il y a un ou deux individus sur le front 

                for j in 0:(taille_front[i][]-1)
                    pop[dec+j].niche_count = Inf
                end
                
            else
            
                for index_glob in 1:length(pop[1].contrainte)
                    
                         # tri des individus par ordre croissant des criteres

                        pop[dec:(dec+taille_front[i][]-1)] = sort(pop[dec:(dec+taille_front[i][]-1)], by = x -> x.contrainte[index_glob])

                        diff_value = pop[dec+taille_front[i][]-1].contrainte[index_glob] - pop[dec].contrainte[index_glob]

                    # calcul du coefficient de surpeuplement seulement en cas
                    # de difference entre les valeurs max et min du critere courant 
                    if diff_value > 0
                    
                        pop[dec].niche_count = Inf
                        pop[dec+taille_front[i][]-1].niche_count = Inf

                        for j in 1:(taille_front[i][]-1)
                            
                            if pop[dec+j].niche_count < Inf
                                pop[dec+j].niche_count += (pop[dec+j+1].contrainte[index_glob] - pop[dec+j-1].contrainte[index_glob])/diff_value
                            end
                        end
                    end
                end
            end # fin else nb individus par front 
        end # fin else individus non realisables 

        dec += taille_front[i][]
    end

    return pop
end


#  ----------------------- fonction Tri_niche -------------------------- */

#  Tri des individu par ordre de front croissant et par indice de densite
#  decroissant

# Fonction Trie_niche
function Trie_niche(pop::Array{Individu}, nb_front::Ref{Int}, taille_front::Vector{Base.RefValue{Int}})
    dec = 1
    for i in 1:nb_front[]
        pop[dec:(dec + taille_front[i][] - 1)] = sort(pop[dec:(dec + taille_front[i][] - 1)],  by = x -> x.niche_count, rev=true) 
        dec += taille_front[i][]
    end
    
    return pop
end


      


# == fonction relatives aux operateurs de variation et de selection ===

# --------------------- fonction Croisement_discret -------------------
#
# Cree deux individus par un croisement discret                         

function Croisement_discret(mate1::Individu, mate2::Individu, nb_param::Int, p0::Float64)
    for i in 1:nb_param
        if Rrandom(0.0, 1.0) < p0
            tmp = mate1.val_param[i]
            mate1.val_param[i] = mate2.val_param[i]
            mate2.val_param[i] = tmp
        end
    end
end

# ----------------------- fonction Croisement_BLX ---------------------
#
# Crée deux individus par croisement de type BLX                 

function Croisement_BLX(mate1::Individu, mate2::Individu, param::Array{TypeParam}, nb_param::Int, dec::Float64)
    for i in 1:nb_param
        delta = Rrandom(0.0 - dec, 1.0 + dec)

        tmp = mate1.val_param[i] + delta * (mate2.val_param[i] - mate1.val_param[i])
        mate2.val_param[i] = mate2.val_param[i] + delta * (mate1.val_param[i] - mate2.val_param[i])
        mate1.val_param[i] = tmp
        
        # Test de dépassement de domaine
        if mate2.val_param[i] > param[i].pmax
            mate2.val_param[i] = param[i].pmax
        end

        if mate2.val_param[i] < param[i].pmin
            mate2.val_param[i] = param[i].pmin
        end

        if mate1.val_param[i] > param[i].pmax
            mate1.val_param[i] = param[i].pmax
        end

        if mate1.val_param[i] < param[i].pmin
            mate1.val_param[i] = param[i].pmin
        end
    end

    return mate1, mate2
end

# ---------------------- fonction Croisement_SBX ---------------------- */

# cree deux individus par croisement binaire simule                 

# On remplace les individu ?
function Croisement_SBX(mate1::Individu, mate2::Individu, param::Vector{TypeParam}, eta::Float64, nb_param::Int)

    u = 0
    tmp1 = 0
    tmp2 = 0
    for i in 1:nb_param
     

	    while true
	   
            while true
                u = Rrandom(0.0,1.0)
                if u != 1
                    break
                end
            end 

			if u <= 0.5
				beta = (2*u)^(1/(eta+1))
			else
				beta = (1/(2*(1-u)))^(1/(eta+1))

                tmp1 = 0.5*((1+beta)*mate1.val_param[i]+
                                             (1-beta)*mate2.val_param[i])
		        tmp2 = 0.5*((1-beta)*mate1.val_param[i]+
                                             (1+beta)*mate2.val_param[i])
            end
	        if !(isnan(tmp1) > 0 || isnan(tmp2) > 0)
                break
            end
        end

        mate2.val_param[i] = tmp2

        mate1.val_param[i] = tmp1

        # test debordement de domaine

		if mate2.val_param[i] > param[i].pmax
            mate2.val_param[i] = param[i].pmax
        end

        if mate2.val_param[i] < param[i].pmin
            mate2.val_param[i] = param[i].pmin
        end

		if mate1.val_param[i] > param[i].pmax
            mate1.val_param[i] = param[i].pmax
        end

        if mate1.val_param[i] < param[i].pmin
           mate1.val_param[i] = param[i].pmin
        end

    end

    return mate1, mate2
end


# -------------------------- fonction BGX_delta -----------------------
#
# Retourne l'amplitude de la perturbation due à la mutation BGX

function BGX_delta(k::Int, discret::Bool)
    delta = 0.0

    if discret
        for j in 1:k
            if Rrandom(0.0, 1.0) < 1.0 / k
                delta += exp(-j * log(2.0))
            end
        end
        if Rrandom(0.0, 1.0) < 0.5
            delta = -delta
        end
    else
        delta = Rrandom(-1.0, 1.0)
        if delta < 0.0
            delta = -exp(delta * k * log(2.0))
        else
            delta = exp(-delta * k * log(2.0))
        end
    end

    return delta
end


# --------------------- fonction Croisement_BGX -----------------------
#
# Crée 1 enfant à partir de deux individus parents par recombinaison BGX

function Croisement_BGX(parent1::Individu, parent2::Individu, enfant::Individu, 
                        param::Vector{TypeParam}, nb_param::Int, k::Int, discrete::Bool)
    norme = 0.0

    for i in 1:nb_param
        norme += (parent1.val_param[i] - parent2.val_param[i])^2 /
                 ((param[i].pmax - param[i].pmin)^2)
    end

    norme = sqrt(norme)

    if norme > 1e-9
        for i in 1:nb_param
            delta = BGX_delta(k, discrete)

            if parent1.front < parent2.front
                enfant.val_param[i] = parent1.val_param[i] +
                    (param[i].pmax - param[i].pmin) * delta / norme *
                    (parent2.val_param[i] - parent1.val_param[i])
            else
                enfant.val_param[i] = parent2.val_param[i] +
                    (param[i].pmax - param[i].pmin) * delta / norme *
                    (parent1.val_param[i] - parent2.val_param[i])
            end

            if enfant.val_param[i] > param[i].pmax
                enfant.val_param[i] = param[i].pmax
            end

            if enfant.val_param[i] < param[i].pmin
                enfant.val_param[i] = param[i].pmin
            end
        end
    else
        enfant = deepcopy(parent1)
    end

    return enfant
end


# -------------------------- fonction Croisement ----------------------

function Croisement(pop::Vector{Individu}, nb_ind::Int, param::Vector{TypeParam},
                    nb_param::Int, k::Int, discrete::Bool, eta::Float64, dec::Float64, type::Int, cross::Cross, current_gen)
    tmp = deepcopy(pop)

    if type == 1  # Croisement de type BGX
        for i in 1:nb_ind
            j = rand(1:nb_ind)
            pop[i] = Croisement_BGX(tmp[i], tmp[j], pop[i], param, nb_param, k, discrete)
        end
    elseif type == 2  # Croisement de type SBX
        for i in 1:2:nb_ind
            pop[i], pop[i+1] = Croisement_SBX(pop[i], pop[i+1], param, eta, nb_param)
        end
    elseif type == 3  # Croisement de type BLX
        for i in 1:2:nb_ind
            pop[i], pop[i+1] = Croisement_BLX(pop[i], pop[i+1], param, nb_param, dec)
        end
    else  # Croisement auto-adaptatif
        for i in 1:2:nb_ind
            if i != nb_ind
                select = Rrandom(0.0, 1.0) < 0.5 ? pop[i].type_cross : pop[i+1].type_cross
            else  # Without this the index i+1 is outside the array range
                select = pop[i].type_cross 
            end

            if select == 1
                pop[i] = Croisement_BGX(tmp[i], tmp[i+1], pop[i],  param, nb_param, k, discrete)
                pop[i+1] = Croisement_BGX(tmp[i], tmp[i+1], pop[i+1], param, nb_param, k, discrete)
                pop[i].type_cross = 1
                pop[i+1].type_cross = 1
                cross.cross_bgx[current_gen] += 1
            elseif select == 2
                pop[i].type_cross = 2
                pop[i+1].type_cross = 2
                pop[i], pop[i+1] = Croisement_SBX(pop[i], pop[i+1], param, eta, nb_param)
                cross.cross_sbx[current_gen] += 1
            elseif select == 3
                pop[i].type_cross = 3
                pop[i+1].type_cross = 3
                pop[i], pop[i+1] = Croisement_BLX(pop[i], pop[i+1], param, nb_param, dec)
                cross.cross_blx[current_gen] += 1
            end
        end
    end

    return pop
end


# function Croisement(pop::Vector{Individu}, nb_ind::Int, param::Vector{TypeParam}, nb_param::Int , k::Int, discrete::Bool, eta::Float64, dec::Float64, type::Int, cross::Cross, current_gen::Int)


#     # Aller verfier que les croisement s'applique binaire


#    if type == 1            # croisement de type BGX 
      
#       	tmp = deepcopy(pop)

#     	for i in 1:nb_ind
#             j = rand(1:nb_ind)
#             pop[i] = Croisement_BGX(tmp[i], tmp[j], pop[i], param, nb_param, k, discrete)
#         end
      
#     elseif type == 2         # croisement de type SBX 
       
#         for i in 1:2:nb_ind
#             pop[i],pop[i+1] = Croisement_SBX(pop[i],pop[i+1],param,eta,nb_param)
#         end
#     elseif type == 3       # croisement de type BLX
      
#          for i in 1:2:nb_ind
#             pop[i],pop[i+1] = Croisement_BLX(pop[i],pop[i+1],param,nb_param,dec)
#          end
      
#     else                   # croisement auto-adaptatif 
#         for i in 1:2:nb_ind
          
#             # selection du croisement a realiser
#             if rand() < 0.5 
#                 select = pop[i].type_cross
#             else
#                 select = pop[i+1].type_cross
#             end

#             # croisement des deux parents en fonction du X-gene

#             if select == 1
          
#                 pop[i] = Croisement_BGX(pop[i], pop[i+1], pop[i], param,nb_param, k, discrete)
#                 pop[i+1] = Croisement_BGX(pop[i], pop[i+1], pop[i+1], param,nb_param, k, discrete)
#                 pop[i].type_cross = 1
#                 pop[i+1].type_cross = 1

#                 cross.cross_bgx[current_gen] += 1
              
#             elseif select == 2

#                 pop[i].type_cross = 2
#                 pop[i+1].type_cross = 2
#                 pop[i], pop[i+1] = Croisement_SBX(pop[i], pop[i+1], param, eta, nb_param)

#                 cross.cross_sbx[current_gen] += 1


#             elseif select == 3
             
#                 pop[i].type_cross = 3
#                 pop[i+1].type_cross = 3
#                 pop[i], pop[i+1] = Croisement_BLX(pop[i], pop[i+1], param, nb_param, dec)

#                 cross.cross_blx[current_gen] += 1
#             end
#         end
#     end
# end

# ----------------------- fonction Mute_individu ----------------------

function Mute_individu(ind::Individu, param::Vector{TypeParam}, nb_param::Int,
                        k::Int, discrete::Bool, pm::Float64)
    for i in 1:nb_param
        if Rrandom(0.0, 1.0) < pm
            delta = BGX_delta(k, discrete)
            ind.val_param[i] += delta * (param[i].pmax - param[i].pmin)

            if ind.val_param[i] > param[i].pmax
                ind.val_param[i] = param[i].pmax
            end

            if ind.val_param[i] < param[i].pmin
                ind.val_param[i] = param[i].pmin
            end
        end
    end
end


# ---------------------- fonction Mute_population ---------------------

function Mute_population(pop::Vector{Individu}, nb_ind::Int, param::Vector{TypeParam},
                         nb_param::Int, k::Int, discrete::Bool, pm::Float64, pm_cross::Float64)
    for i in 1:nb_ind
        Mute_individu(pop[i], param, nb_param, k, discrete, pm)
    end

    # Mutation des gènes liés au croisement
    for i in 1:nb_ind
        if Rrandom(0.0, 1.0) < pm_cross
            pop[i].type_cross = rand(1:3)
        end
    end

    return pop
end

# ------------------------ fonction Selection -------------------------

function Selection!(pop::Vector{Individu}, nb_ind::Int)

    k, j = 0, 0
    # Pour chaque individu de la population tampon on va le remplacer par un individu séléctionné dans la pop archivée.
    for i in 1:nb_ind

        # On séléctionne 2 indice pour organiser un combat, on recommence tant qu'il ne sont pas différents
        j = rand(1:nb_ind)
        while true
            k=rand(1:nb_ind)
            if k!=j
                break
            end
        end

        # Si ind_j domine ind_k (pris dans la pop archivée), on stock ind_j dans la pop tampon car il a gagné
        if (Dominance_individu(pop[j],pop[k]))
            pop[nb_ind+i] = deepcopy(pop[j])
        # Sinon si ind_k domine ind_j (pris dans la pop archivée), on stock ind_k dans la pop tampon car il a gagné
        elseif (Dominance_individu(pop[k],pop[j]))
            pop[nb_ind+i] = deepcopy(pop[k])
        # Sinon si ind_j à un meilleur indice de nichage il gagne
        elseif (pop[j].niche_count > pop[k].niche_count)
                pop[nb_ind+i] = deepcopy(pop[j])
        else
        # Sinon c'est ind_k qui gagne
            pop[nb_ind+i] = deepcopy(pop[k])
        end
    end
end



function affiche_individu(fichier::IO, ind::Individu, nb_param::Int, nb_crit::Int, nb_cont::Int)
    for i in 1:nb_param
        println(fichier, ind.val_param[i])
    end
    
    for i in 1:nb_cont
        println(fichier, ind.contrainte[i])
        println(fichier, ind.realisable)
    end
    
    for i in 1:nb_crit
        println(fichier, ind.critere[i])
    end
    
    println(fichier, ind.front)
    println(fichier, ind.niche_count)
end

function Affiche_population(fichier::IO, pop::Vector{Individu}, nb_ind::Int, nb_param::Int, nb_crit::Int, nb_cont::Int)
    for i in 1:nb_ind
        affiche_individu(fichier, pop[i], nb_param, nb_crit, nb_cont)
    end
end





# # Constantes de dimensionnement des tableaux
# const NMAX_PARAM = 20  # nombre maximal de parametres
# const NMAX_CRIT = 10   # nombre maximal de criteres
# const NMAX_CONT = 10   # nombre maximal de contraintes
# const NMAX_IND = 1000   # taille maximale de la population
# const NMAX_POP = 2 * NMAX_IND  # taille maximale de la population+archive

# using Random
# # Caractéristiques du problème d'optimisation

# nb_crit = 6
# nb_cont = 2




# nb_front = Ref(0)
# taille_front = [Ref(0) for _ in 1:NMAX_IND]



# lb_var = [-10.0, -10.0]
# ub_var = [10.0, 10.0]
# nb_param = length(ub_var)
# param = [TypeParam(0.0, 0.0) for _ in 1:nb_param]


# @assert(length(ub_var) == length(lb_var))
# for i in 1:nb_param
#     param[i] = TypeParam(lb_var[i], ub_var[i])
# end


# nb_gene = 500         # nombre de generation max 
# nb_ind = 200     # nombre d'individus - taille de la population
# nb_pop = 2*nb_ind      # taille globale avec archive

# pm = 1.0/nb_param      # taux de mutation - defaut: 1.0/nb_param
# pm_cross = 0.05        # taux de mutation du X-gene - defaut: 5%

# type= 4                # type de croisement - defaut : 4
#                         # 1 = BGX
#                         # 2 = SBX
#                         # 3 = BLX
#                         # autre = autoadaptatif #
# dec = 0.5              # parametre du BLX - defaut: 0.5                    
# eta = 1.0              # parametre du SBX - defaut: 1.0
# k = 16                 # parametre du BGX - defaut: 16
# discrete = true        # parametre de mutation - defaut VRAI   

  
# # Initialisation des variables globales
    
# cross = Cross(zeros(nb_gene),zeros(nb_gene),zeros(nb_gene))





# function f_eval(params::Vector{Float64})
    
#     max_dist_cont = sqrt((params[1]-5)^2 + (params[2]-5)^2) - 5
#     min_dist_cont =  -(sqrt((params[1]-5)^2 + (params[2]-5)^2) - 4)

#     contraintes = [max_dist_cont, min_dist_cont]

#     max_dist_obj = -sqrt((params[1]-5)^2 + (params[2]-5)^2) 
#     min_dist_obj = sqrt((params[1]-5)^2 + (params[2]-5)^2)

#     criteres = vcat([max_dist_obj, min_dist_obj], [params[1], params[2]])


#     return contraintes, criteres
# end



# using PyPlot
# pygui(true)


# # Initialisation du generateur de nombres aleatoires
# Randomize()

   
# println("generation = ", 0)  #  affichage des generations a l'ecran


# # seems ok
# pop = Initialise_population!(param, nb_param, nb_pop, f_eval)

# for i in 1:nb_gene
#     println("generation = ",i)  #  affichage des generations a l'ecran

#     Calcule_dominance!(pop, nb_pop, nb_front, taille_front)

#     pop = Affecte_niche(pop, nb_pop, nb_front, taille_front)

#     pop = Trie_niche(pop, nb_front, taille_front)

#     # On en est la

#     Selection!(pop, nb_ind, nb_crit, nb_cont)


#     pop[(nb_ind+1):nb_pop] .= Croisement(pop[(nb_ind+1):nb_pop], nb_ind, param, nb_param, k, discrete, eta, dec, type, cross, i)

#     pop[(nb_ind+1):nb_pop] .= Mute_population(pop[(nb_ind+1):nb_pop], nb_ind, param, nb_param, k, discrete, pm, pm_cross)
   
#     for j in 1:nb_ind
#         pop[nb_ind+j].critere , pop[nb_ind+j].contrainte, pop[nb_ind+j].realisable = Calcule_criteres(pop[nb_ind+j].val_param, f_eval)   
#     end
# end


# Calcule_dominance!(pop,nb_pop, nb_front, taille_front)
# pop = Affecte_niche(pop, nb_pop, nb_front, taille_front)
# pop = Trie_niche(pop, nb_front, taille_front)



# ############## Croisement evolution
# figure("Croisement")
# plot(1:nb_gene, cross.cross_bgx, label="BGX")
# plot(1:nb_gene, cross.cross_blx, label="BLX")
# plot(1:nb_gene, cross.cross_sbx, label="SBX")
# legend()


# ################# critère
# figure("front Julia")

# scatter([a.critere[1] for a in  pop[1:nb_ind]],[a.critere[2] for a in pop[1:nb_ind]])





# ################# critère
# figure("Param Julia")


# x = [x for x in 0:0.1:10]
# y = sqrt.((25 .- (x .- 5).^2)) .+ 5
# y2 = -sqrt.((25 .- (x .- 5).^2)) .+ 5

# PyPlot.scatter(repeat(x, 2), vcat(y, y2))
# PyPlot.scatter(repeat(x, 2), vcat(y, y2))


# x = [x for x in 1:0.1:9]
# y = sqrt.((16 .- (x .- 5).^2)) .+ 5
# y2 = -sqrt.((16 .- (x .- 5).^2)) .+ 5

# PyPlot.scatter(repeat(x, 2), vcat(y, y2))
# PyPlot.scatter(repeat(x, 2), vcat(y, y2))

# scatter([a.val_param[1] for a in  pop[1:nb_ind]],[a.val_param[2] for a in pop[1:nb_ind]])

 



# #fichier = open("RESGEN.RES", "w")
# #Affiche_population(fichier, pop, nb_ind, nb_param, nb_crit, nb_cont)
# #close(fichier)




# # Faire une fonction de test

# if false
#     using FileIO, CSV, DataFrames, Dates, JLD2

#     file = CSV.File(joinpath(pwd(), "src", "optimization", "designer", "metaheuristic","NSGAII","RESGEN_C.RES"))

#     data = DataFrames.DataFrame(file)
#     figure("front C")

#     scatter(data.critere0 ,data.critere1)
# end
