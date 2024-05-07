function optimizeMetaheuristic(f::Function,
				  lb, ub,
				  method::NSGAII;
				  options = MetaResultOptions())


        nb_param = method.nb_param
        nb_pop = method.nb_pop
        nb_ind = method.nb_ind
        nb_gene = options.iterations

        nb_front = Ref(0)
        taille_front = [Ref(0) for _ in 1:nb_pop]

        @assert(length(ub) == length(lb))
        param = [TypeParam(0.0, 0.0) for _ in 1:nb_param]

        for i in 1:nb_param
            param[i] = TypeParam(lb[i], ub[i])
        end
        
        cross = Cross(zeros(nb_gene),zeros(nb_gene),zeros(nb_gene))

        # Initialisation du generateur de nombres aleatoires
        Randomize()
        
        println("generation = ", 0)  #  affichage des generations a l'ecran

        pop = Initialise_population!(param, nb_param, nb_pop, f)

        # On grade une trace de la population realisable
        real_track = []

        @showprogress for i in 1:nb_gene
            println("generation = ",i)  #  affichage des generations a l'ecran

            Calcule_dominance!(pop, nb_pop, nb_front, taille_front)

            pop = Affecte_niche(pop, nb_pop, nb_front, taille_front)

            pop = Trie_niche(pop, nb_front, taille_front)

            # On stock les éléments réalisables
            real_track = vcat(real_track, filter!(x -> x.realisable, pop[1:nb_ind]))

            Selection!(pop, nb_ind)

            pop[(nb_ind+1):nb_pop] .= Croisement(pop[(nb_ind+1):nb_pop], nb_ind, param, nb_param, method.k, method.discrete, method.eta, method.dec, 4, cross, i) # 4 is for auto-adaptativ

            pop[(nb_ind+1):nb_pop] .= Mute_population(pop[(nb_ind+1):nb_pop], nb_ind, param, nb_param, method.k, method.discrete, method.pm, method.pm_cross)
        
            #Threads.@threads
            for j in 1:nb_ind
                pop[nb_ind+j].critere , pop[nb_ind+j].contrainte, pop[nb_ind+j].realisable = Calcule_criteres(pop[nb_ind+j].val_param, f)   
            end
        end

        Calcule_dominance!(pop,nb_pop, nb_front, taille_front)
        pop = Affecte_niche(pop, nb_pop, nb_front, taille_front)
        pop = Trie_niche(pop, nb_front, taille_front)

        SA = compute_SA(real_track)

        # Les resultats sont une structure contenant : 
        # - La method
        # - Le tracking pour le calcul de SA
        # - La pop
        # - Les indices de sobol de premier ordre et les correlation de pearson

        plot_results(pop[1:nb_ind], cross)
        
        return NSGAIIResults(method, real_track, pop[1:nb_ind], SA)
end
