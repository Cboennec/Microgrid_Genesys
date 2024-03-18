// Code NSGA-II avec Croisement autoadaptatif - Couplage Matlab-Simulink
// Bruno Sareni, LAPLACE - Mise a jour 2009
// References :  
// Deb et al, A fast Elitist Multiobjective Genetic Algorithm : NSGA-II
// IEEE Trans. on Evolutionary Computation, Vol 6, N°2, 2002
// Sareni et al, Recombination and Self-Adaptation in Multiobjective 
// Genetic Algorithms, LNCS, Vol 2936, 2004

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/* === constantes necessaires pour la redefinition d'un type booleen === */

#define VRAI 1
#define FAUX 0

/* ============ constantes de dimensionnement des tableaux ============= */

#define NMAX_PARAM  20          // nombre maximal de parametres    

#define NMAX_CRIT 10            // nombre maximal de criteres       

#define NMAX_CONT 10             // nombre maximal d'individus      

#define NMAX_IND 500            // taille maximale de la population  

#define NMAX_POP 2*NMAX_IND     // taille maximale de la population+archive
 										               	       
/* ========================== variables globales ======================= */

int nb_eval;                // nombre total d'evaluations   
int index_glob;             // indice global
int cross_bgx;              // nombre de croisements BGX
int cross_sbx;              // nombre de croisements SBX
int cross_blx;              // nombre de croisements BLX_
int nb_real;                // nombre total d'individus realisables evalues

/* ======================= definition des types ======================== */

/* -------------------- definition du type booleen --------------------- */

typedef char BOOLEEN;

/* --------------------- definition du type individu ------------------- */

typedef struct individu            // caracteristiques des individus
 {
   int front;                              // front d'appartenance
   double niche_count;                     // indice de densite
   double val_param[NMAX_PARAM];           // parametres
   double contrainte[NMAX_CONT];           // contraintes
   double critere[NMAX_CRIT];              // criteres
   int type_cross;                         // X-gene : type de croisement
   BOOLEEN realisable;                     // = 1 si realisable
 } INDIVIDU;

/* ------------------ definition du type type_param -------------------- */

typedef struct type_param          // contraintes de domaines
 {
   double pmin,pmax;
 } TYPE_PARAM;

/* ======== fonctions pour la generation des nombres aleatoires ======== */
 
/* ----------------------- fonction Randomize -------------------------- */
 
// initialise les generateurs de nombre aleatoires
 
 void Randomize (void)      
 
 {
   srand ( (unsigned)time(NULL) );

 }

/* -------------------------- fonction Brandom ------------------------- */

// retourne une valeur valeur aleatoire booleene                        

BOOLEEN Brandom (void)

 {
   return (BOOLEEN) ((double)rand() / ((double)RAND_MAX + 1) * 2);
 }

/* -------------------------- fonction Rrandom ------------------------- */

// retourne un reel pris au hasard dans l'intervalle [min, max]           

double Rrandom (double min, double max)

 {
   return ( rand() * (max - min) / RAND_MAX + min );
 }

/* --------------------------- fonction Erandom ------------------------ */

// retourne un entier pris a hasard dans l'intervalle [min, max]          

int Erandom (int min, int max)

 {
   int res;
   res =  (int) (Rrandom(0.0,1.0) * (max - min + 1) + min);
   if (res > max)
	res = max;
   return res;
 }

/* ====== definition des fonctions de tri en argument pour qsort ======= */ 

/* --------------------- fonction Crit_Compare ------------------------- */

// fonction de comparaison pour le tri par critere croissant 
 
int Crit_compare (const INDIVIDU *n1, const INDIVIDU *n2)

 {
   int resultat;

   if (n1 -> critere[index_glob] > n2-> critere[index_glob])
       resultat = 1;
   else if (n1 -> critere[index_glob] < n2 -> critere[index_glob])
           resultat = -1;
        else resultat = 0;

   return (resultat);
 }

/* ---------------------- fonction Cont_Compare ------------------------ */

// fonction de comparaison pour le tri par contrainte croissant
  
int Cont_compare (const INDIVIDU *n1, const INDIVIDU *n2)

 {
   int resultat;

   if (n1 -> contrainte[index_glob] > n2-> contrainte[index_glob])
       resultat = 1;
   else if (n1 -> contrainte[index_glob] < n2 -> contrainte[index_glob])
           resultat = -1;
        else resultat = 0;

   return (resultat);
 }

/* --------------------- fonction Niche_Compare ------------------------ */ 
 
// fonction de comparaison pour le tri par indice de niche decroissant

int Niche_compare (const INDIVIDU *n1, const INDIVIDU *n2)

 {
   int resultat;

   if (n1 -> niche_count < n2-> niche_count)
       resultat = 1;
   else if (n1 -> niche_count > n2 -> niche_count)
           resultat = -1;
        else resultat = 0;

   return (resultat);
 } 
 
/* Calcul des contraintes et des criteres - initialisation des individus */ 

/* -------------------- fonction Calcule_criteres ---------------------- */
 
// Calcul des critères et des contraintes associes a un individu
// Les contraintes doivent etre definies en termes de minimisation 
// Elles sont affectees a zero si elles sont verifiees 
// Elles sont d'autant plus positives qu'elles sont violees
// ind -> realisable doit etre affecte a FAUX lorsque une contrainte est 
// violee
   
void Calcule_criteres (INDIVIDU *ind, int nb_crit, int nb_param, 
                       int nb_cont)
 {
    int i;
    
    // initialisations
    
    nb_eval++;                  // incrementation du compteur d'evaluations
    ind -> realisable = VRAI;   // par defaut realisable = VRAI

    // calcul des contraintes
    
    ind -> contrainte[0] = 4 - ( pow(ind -> val_param[0] - 5,2.0) +
                                 pow(ind -> val_param[1] - 5,2.0));
    ind -> contrainte[1] = pow(ind -> val_param[0] - 5,2.0) +
                           pow(ind -> val_param[1] - 5,2.0) - 9;
    
    // Annulation des contraintes negatives c.a.d verifiees  
    
    for (i = 0; i < nb_cont; i++)
     if (ind -> contrainte[i] < 0)
         ind -> contrainte[i] = 0;
    
   // Affectation de la faisabilite 
    
    for (i = 0; i < nb_cont; i++)
     if (ind -> contrainte[i] > 0)
       {
         ind -> realisable = FAUX;
         break;
       }
    
    // Incrementation du compteur de faisabilite
    
    if (ind -> realisable)
        nb_real++;

    // Calcul des criteres en cas de faisabilite
    
    if (ind -> realisable)
    {  
        ind -> critere[0] = ind -> val_param[0];
        ind -> critere[1] = ind -> val_param[1];
    }   
       
    return;
 }

/* ------------------- fonction Initialise_individu -------------------- */

// Initialise de facon uniformement aleatoire les valeurs des parametres 
// à l'interieur des contraintes de domaine definissant l'espace de 
// recherche.
// Initilise de facon uniformement aleatoire les valeurs du X-gene codant
// le type de croisement a appliquer.
   
void Initialise_individu (INDIVIDU *ind, TYPE_PARAM param[], int nb_param,
                          int nb_crit, int nb_cont)

 {
    int i;

    for (i = 0; i < nb_param; i++)
      {
        ind -> val_param[i] = Rrandom(param[i].pmin,param[i].pmax);
        ind -> type_cross = Erandom(1,3);
      }
      
      
    Calcule_criteres(ind,nb_crit,nb_param,nb_cont);
  }

/* ------------------ fonction Initialise_population ------------------- */  

// Initialise la population d'individus
  
void Initialise_population(INDIVIDU pop[], TYPE_PARAM param[], int nb_param,
                           int nb_crit, int nb_ind, int nb_cont)

 {

    int i;

    for (i = 0; i < nb_ind; i++)
      Initialise_individu(&pop[i],param,nb_param,nb_crit,nb_cont);

  }
  
/* ===== fonctions necessaires a la dertermination de la dominance ===== */ 

/* -------------------- fonction Allocation_matrice -------------------- */

 // Allocation dynamique d'une matrice de dimension nb_ind * nb_ind 
 
int ** Allocation_matrice ( int nb_ind )

{

int i;
int **matrice ;

matrice = (int **) malloc(nb_ind*sizeof(int *));

for (i=0;i<nb_ind;i++)
   matrice[i] = (int *) malloc (nb_ind*sizeof(int));

return (matrice);

}

/* ---------------------- fonction Libere_matrice ---------------------- */

// Liberation de la memoire pour une matrice de dimension nb_ind * nb_ind     

void Libere_matrice(int **matrice ,int nb_ind)

{
int i;

for (i=0;i<nb_ind;i++)
   free (matrice[i]);

free (matrice);
}

/* -------------------- fonction Dominance_individu -------------------- */

// Etablissement de la dominance entre deux individus
// Un individu realisable domine un individu non realisable
// Si les 2 individus sont non realisables, la dominance est etablie dans
// l'espace des contraintes
// Si les 2 individus sont realisables, la dominance est etablie dans 
// l'espace des criteres

int Dominance_individu(INDIVIDU *ind1 , INDIVIDU *ind2 , int nb_crit,
					   int nb_cont)
{

  int i;

   // ind1 realisable / ind2 non realisable

  if (ind1 -> realisable == VRAI  && ind2 -> realisable == FAUX)
	 return (1);
  else
   // ind1 non realisable / ind2 realisable 
     if (ind1 -> realisable == FAUX  && ind2 -> realisable == VRAI)
	   return (0);
  else
   // ind1 et ind2 non realisables 
     if (ind1 -> realisable == FAUX  && ind2 -> realisable == FAUX)
	 {
	   for (i=0;i<nb_cont;i++)
	   {
         if (ind1->contrainte[i] > ind2->contrainte[i])

         return (0);
	   }
  	   return (1);
	 }
   else
	// ind1 et ind2 realisables
   {
	 for (i=0;i<nb_crit;i++)
	 {
       if (ind1->critere[i] > ind2->critere[i])

       return (0);
	 }
	 return (1);
   }
}

/* -------------------- fonction Initialisation_MDSL ------------------- */

// Initialisation matrice de dominance au sens large         

void Initialisation_MDSL ( int ** matrice , int nb_ind )

{
int i;
int j;


  for ( i = 0; i < nb_ind ; i++)

    for ( j =0; j < nb_ind ; j++)

      matrice[i][j] = 1 ;
}

/* -------------------- fonction Construction_MDSL --------------------- */

// Construction de la matrice de dominance au sens large             

void Construction_MDSL (INDIVIDU pop[], int **matrice_SL ,int nb_ind,
			int nb_crit,int nb_cont)

{
int i ;
int j ;
int egalite ;

Initialisation_MDSL (matrice_SL , nb_ind);

for (i = 0; i < nb_ind; i++)
 {
  for ( j = 0; j < (nb_ind-1-i); j++)
    {
   egalite = Dominance_individu (&pop[i], &pop[1+i+j],nb_crit,nb_cont);

   if (egalite == 0 )

    matrice_SL[i][i+j+1] = 0 ;

     }
  }

for (i = 0; i < nb_ind; i++)
 {
  for ( j = 0; j < (nb_ind-1-i); j++)
    {
   egalite = Dominance_individu(&pop[1+i+j],&pop[i],nb_crit,nb_cont);

   if (egalite == 0 )

    matrice_SL[i+j+1][i] = 0 ;
     }
  }
}

/* -------------------- fonction Construction_MDSS --------------------- */

// Construction de la matrice de dominance au sens stricte

void Construction_MDSS (INDIVIDU pop[],int **matrice_SS, int nb_ind,
			int nb_crit,int nb_cont)
{
  int i ;
  int j ;
  int **matrice_SL;

  matrice_SL = Allocation_matrice(nb_ind);

  Construction_MDSL(pop,matrice_SL,nb_ind,nb_crit,nb_cont);

  for (i=0;i<nb_ind;i++)

    for(j=0;j<nb_ind;j++)

       matrice_SS[i][j] = matrice_SL[i][j]*(1 - matrice_SL[j][i]);

  Libere_matrice(matrice_SL,nb_ind);
 }

/* ---------------- fonction Echange_ligne_colonne --------------------- */

// Echanges deux lignes et permute les colonnes correspondantes
  
void Echange_ligne_colonne (int i1, int i2 , int nb_ind,
									 int ** matrice_SS)

{
  int k,tmp;

  // permute lignes

  for (k = 0; k < nb_ind; k++)
   {
    tmp = matrice_SS [i1][k];
    matrice_SS[i1][k] = matrice_SS[i2][k];
    matrice_SS[i2][k] = tmp;
   }

  // permute colonnes

  for (k = 0; k < nb_ind; k++)
   {
    tmp = matrice_SS [k][i1];
    matrice_SS[k][i1] = matrice_SS[k][i2];
    matrice_SS[k][i2] = tmp;
   }
}

/* --------------------- fonction Determine_fronts --------------------- */

// Determine les fronts associes a chaque individu
// front = 0 = individu non domine

void Determine_fronts (INDIVIDU pop[], int nb_ind, int *frt, 
                       int taille_front[], int ** matrice_SS)

{
  int dec_front, dec, i,j;
  INDIVIDU tmp;

  *frt = 0;         // front courant 
  dec = 0;          // decalage colonne dans la matrice 

  do
  {

    dec_front = dec;
    taille_front[*frt] = 0;

    for (j = dec_front; j < nb_ind; j++)
      {
         for (i = dec_front; i < nb_ind; i++)
           if (matrice_SS[i][j]==1)
             break;

         if (i == nb_ind)        // individu non domine sur ce front 
           {
             taille_front[*frt]++;
             if (j!=dec)
               {
                  Echange_ligne_colonne (dec,j,nb_ind,matrice_SS);
                  tmp = pop[dec];
                  pop[dec] = pop[j];
                  pop[j] = tmp;
               }
             pop[dec].front = *frt;
             dec ++;
           }
      }

    (*frt)++;
  } while (dec < nb_ind);
}

/* --------------------- fonction calcul_dominance --------------------- */

// affecte les front a tous les individus de la population

void Calcule_dominance (INDIVIDU pop[],int nb_ind, int nb_crit, 
                        int nb_cont,int *nb_front,int taille_front[])
 {
    int **matrice_DSS;

    matrice_DSS = Allocation_matrice (nb_ind);

    Construction_MDSS (pop,matrice_DSS,nb_ind,nb_crit,nb_cont);

    Determine_fronts (pop,nb_ind,nb_front,taille_front,matrice_DSS);

    Libere_matrice (matrice_DSS, nb_ind);

    return;
  }


/* --------------------- fonction Affecte_niche ------------------------ */

// affecte a tous les individus un indice de densite par rapport a leur 
// position sur le front auquel ils appartiennent. Les individus les moins
// representes (les extremes) recoivent un indice infini. Plus densite au 
//  voisinage d'un individu est forte, plus l'indice est petit.
  
void Affecte_niche (INDIVIDU pop[],int nb_ind,int nb_crit, int nb_cont,
						  int nb_front,int taille_front[])
 {
    int dec,i,j;
    double diff_value;

    dec = 0;     // repere place sur le front courant 

    for (j = 0; j < nb_ind; j++)  // initialisation des niches
     pop[j].niche_count = 0;

    for (i = 0; i < nb_front; i++)
     {
       if (pop[dec].realisable == VRAI)
         {
           // nichage en fonction des criteres

           if (taille_front[i]<3)
             {
               // il y a un ou deux individus sur le front

               for (j = 0; j < taille_front[i]; j++)
                  pop[dec+j].niche_count = DBL_MAX;
             }
           else
             {
               for (index_glob = 0; index_glob < nb_crit; index_glob++)
                 {
                    // tri des individus par ordre croissant des criteres  

                    qsort(&pop[dec],taille_front[i],sizeof(INDIVIDU),
                    (int(*) (const void*, const void*))Crit_compare);

                     diff_value = 
                      pop[dec+taille_front[i]-1].critere[index_glob]-
                      pop[dec].critere[index_glob];

                 // calcul du coefficient de surpeuplement seulement en cas
                 // de difference entre les valeurs max et min du critere
                 // courant 

                    if (diff_value > 0)
                     {
                       pop[dec].niche_count = DBL_MAX;
                       pop[dec+taille_front[i]-1].niche_count = DBL_MAX;

                       for (j=1; j < taille_front[i]-1; j++)
                         {
                           if (pop[dec+j].niche_count < DBL_MAX)
                           {
                           pop[dec+j].niche_count += 
                           (pop[dec+j+1].critere[index_glob]
                           -pop[dec+j-1].critere[index_glob])/diff_value;
                           }
                         }
                     }
                 }
             } // fin else nb individus par front 
         }  // fin if individus realisable 
       else
        {
           // nichage en fonction des contraintes 

           if (taille_front[i]<3)
             {
               // il y a un ou deux individus sur le front 

               for (j = 0; j < taille_front[i]; j++)
                  pop[dec+j].niche_count = DBL_MAX;
             }
           else
             {
               for (index_glob = 0; index_glob < nb_cont; index_glob++)
                 {
                    // tri des individus par ordre croissant des criteres

                    qsort(&pop[dec],taille_front[i],sizeof(INDIVIDU),
                    (int(*) (const void*, const void*))Cont_compare);

                    diff_value = 
                     pop[dec+taille_front[i]-1].contrainte[index_glob]
                    -pop[dec].contrainte[index_glob];

                 // calcul du coefficient de surpeuplement seulement en cas
                 // de difference entre les valeurs max et min du critere
                 // courant 

                    if (diff_value > 0)
                     {
                       pop[dec].niche_count = DBL_MAX;
                       pop[dec+taille_front[i]-1].niche_count = DBL_MAX;

                       for (j=1; j < taille_front[i]-1; j++)
                         {
                           if (pop[dec+j].niche_count < DBL_MAX)
                           {
                           pop[dec+j].niche_count += 
                           (pop[dec+j+1].contrainte[index_glob]-
                           pop[dec+j-1].contrainte[index_glob])/diff_value;
                           }
                         }
                     }
                 }
             } // fin else nb individus par front 
        } // fin else individus non realisables 

       dec += taille_front[i];
     }
 }

/* ----------------------- fonction Tri_niche -------------------------- */

// Tri des individu par ordre de front croissant et par indice de densite
// decroissant

void Trie_niche (INDIVIDU pop[],int nb_front,int taille_front[])
 {

   int i, dec;

   dec = 0;
   for (i = 0; i < nb_front; i++)
    {
      qsort(&pop[dec],taille_front[i],sizeof(INDIVIDU),
                    (int(*) (const void*, const void*))Niche_compare);
      dec += taille_front[i];
    }
 }
      
/* == fonction relatives aux operateurs de variation et de selection === */

/* --------------------- fonction Croisement_discret ------------------- */

// cree deux individus par un croisement discret                         

void Croisement_discret (INDIVIDU *mate1, INDIVIDU *mate2, int nb_param,
                         double p0)
 {
   int i;
   double tmp;

   for (i = 0; i < nb_param; i++)
       if (Rrandom(0.0,1.0) < p0)
            {
               tmp = mate1 -> val_param[i];
               mate1 -> val_param[i] = mate2 -> val_param[i];
               mate2 -> val_param[i] = tmp;
            }
 }

/* ----------------------- fonction Croisement_BLX --------------------- */

// cree deux individus par croisement de type BLX                 

void Croisement_BLX (INDIVIDU *mate1, INDIVIDU *mate2,
                              TYPE_PARAM param[],int nb_param, double dec)
 {
   int i;
   double delta,tmp;


   for (i = 0; i < nb_param; i++)
     {
        delta = Rrandom (0.0-dec,1.0+dec);

        tmp = mate1 -> val_param[i] + delta *
                        (mate2 -> val_param[i] - mate1 -> val_param[i]);

        mate2 -> val_param[i] = mate2 -> val_param[i] + delta *
                        (mate1 -> val_param[i] - mate2 -> val_param[i]);

        mate1 -> val_param[i] = tmp;
        
        // test debordement de domaine

		if (mate2 -> val_param[i] > param[i].pmax)
            mate2 -> val_param[i] = param[i].pmax;

        if (mate2 -> val_param[i] < param[i].pmin)
            mate2 -> val_param[i] = param[i].pmin;

		if (mate1 -> val_param[i] > param[i].pmax)
            mate1 -> val_param[i] = param[i].pmax;

        if (mate1 -> val_param[i] < param[i].pmin)
            mate1 -> val_param[i] = param[i].pmin;
     }
 }

/* ---------------------- fonction Croisement_SBX ---------------------- */

// cree deux individus par croisement binaire simule                 

void Croisement_SBX(INDIVIDU *mate1, INDIVIDU *mate2,
                             TYPE_PARAM param[], double eta, int nb_param)

 {
   int i;
   double tmp1, tmp2, beta, u;


   for (i = 0; i < nb_param; i++)
     {

	   do
	   {
         do
			u = Rrandom(0.0,1.0);
         while (u==1);

			if (u<= 0.5)
				beta = pow(2*u,1/(eta+1));
			else
				beta = pow(1/(2*(1-u)),1/(eta+1));

                tmp1 = 0.5*((1+beta)*mate1->val_param[i]+
                                             (1-beta)*mate2->val_param[i]);
		        tmp2 = 0.5*((1-beta)*mate1->val_param[i]+
                                             (1+beta)*mate2->val_param[i]);

	   } while (_isnan(tmp1)>0 || _isnan(tmp2)>0);

        mate2 -> val_param[i] = tmp2;

        mate1 -> val_param[i] = tmp1;

        // test debordement de domaine

		if (mate2 -> val_param[i] > param[i].pmax)
            mate2 -> val_param[i] = param[i].pmax;

        if (mate2 -> val_param[i] < param[i].pmin)
            mate2 -> val_param[i] = param[i].pmin;

		if (mate1 -> val_param[i] > param[i].pmax)
            mate1 -> val_param[i] = param[i].pmax;

        if (mate1 -> val_param[i] < param[i].pmin)
           mate1 -> val_param[i] = param[i].pmin;

     }
 }

/* -------------------------- fonction BGX_delta ----------------------- */

// retourne l'amplitude de la perturbation due a la mutation BGX              

double BGX_delta (int k, BOOLEEN discret)

 {
   int j;
   double delta;

   if (discret)
      {
         delta = 0;
         for (j = 0; j < k; j++)
            if (Rrandom(0.0,1.0) < 1.0 / ((double)k))
               delta += exp(-j*log (2.0));
         if (Rrandom(0.0,1.0) < 0.5)
               delta = -delta;
      }
    else
      {
         delta = Rrandom(-1.0,1.0);
         if (delta < 0.0)
             delta = -exp(delta*k*log(2.0));
         else
             delta = exp(-delta*k*log(2.0));
      }

    return delta;

 }

/* --------------------- fonction Croisement_BGX ----------------------- */

// cree 1 enfant a partir de deux individus parents par recombinaison BGX

void Croisement_BGX (INDIVIDU *parent1,INDIVIDU *parent2,INDIVIDU *enfant,
                 TYPE_PARAM param[], int nb_param, int k, BOOLEEN discrete)

 {
   int i;
   double delta,norme;


   norme = 0;
   for (i = 0; i < nb_param; i++)
     norme += (parent1 -> val_param[i] - parent2 -> val_param[i]) *
              (parent1 -> val_param[i] - parent2 -> val_param[i])
              /(param[i].pmax-param[i].pmin)/(param[i].pmax-param[i].pmin);

   norme = sqrt(norme);

   if (norme > 1e-9)
     for (i = 0; i < nb_param; i++)
       {

         delta = BGX_delta(k,discrete);

         if (parent1 -> front < parent2 -> front)
                enfant -> val_param[i] = parent1 -> val_param[i] +
                    (param[i].pmax - param[i].pmin) * delta / norme *
                    (parent2 -> val_param[i] - parent1 -> val_param[i]);
         else
                enfant -> val_param[i] = parent2 -> val_param[i] +
                    (param[i].pmax - param[i].pmin) * delta / norme *
                    (parent1 -> val_param[i] - parent2 -> val_param[i]);

         if (enfant -> val_param[i] > param[i].pmax)
             enfant -> val_param[i] = param[i].pmax;

         if (enfant -> val_param[i] < param[i].pmin)
             enfant -> val_param[i] = param[i].pmin;
       }
   else
       *enfant = *parent1;
 }

/* -------------------------- fonction Croisement ---------------------- */

void Croisement (INDIVIDU pop[], int nb_ind, TYPE_PARAM param[],
    int nb_param,int k, BOOLEEN discrete, double eta, double dec, int type)

{
	int i,j,select;

	INDIVIDU tmp[NMAX_IND];

   if (type == 1)            // croisement de type BGX 
      {
      	for (i = 0; i<nb_ind; i++)
	        tmp[i] = pop[i];

    	   for (i = 0; i<nb_ind; i++)
	       {
	         j = Erandom(0,nb_ind-1);
	         Croisement_BGX (&tmp[i],&tmp[j],&pop[i],param,nb_param,
                             k,discrete);
	       }
      }
   else
     if (type == 2)         // croisement de type SBX 
       {
         for (i=0; i<nb_ind; i+=2)
           Croisement_SBX(&pop[i],&pop[i+1],param,eta,nb_param);
       }
   else
     if (type == 3)        // croisement de type BLX
      {
         for (i=0; i<nb_ind; i+=2)
           Croisement_BLX (&pop[i],&pop[i+1],param,nb_param,dec);
      }
  else                    // croisement auto-adaptatif 
     {
        for (i = 0; i< nb_ind; i+=2)
          {
                         // selection du croisement a realiser

				if (Rrandom(0.0,1.0)<0.5)
					select = pop[i].type_cross;
				else
               select = pop[i+1].type_cross;

            // croisement des deux parents en fonction du X-gene

            if (select == 1)
              {
                tmp[0] = pop[i];
                tmp[1] = pop[i+1];
                Croisement_BGX (&tmp[0],&tmp[1],&pop[i],
                                param,nb_param,k,discrete);
                Croisement_BGX (&tmp[0],&tmp[1],&pop[i+1],
                                param,nb_param,k,discrete);
                pop[i].type_cross = 1;
                pop[i+1].type_cross = 1;
                cross_bgx++;
              }
           else if (select == 2)
             {
                pop[i].type_cross = 2;
                pop[i+1].type_cross = 2;
                Croisement_SBX(&pop[i],&pop[i+1],param,eta,nb_param);
                cross_sbx++;
             }

           else if (select == 3)
             {
              pop[i].type_cross = 3;
              pop[i+1].type_cross = 3;
              Croisement_BLX(&pop[i],&pop[i+1],param,nb_param,dec);
              cross_blx++;
             }
         }
     }
}

/* ----------------------- fonction Mute_individu ---------------------- */

// cree une perturbation avec une probabilite pm sur un gene d'un individu                                                                

void Mute_individu (INDIVIDU *ind, TYPE_PARAM param[], int nb_param,
                      int k, BOOLEEN discrete, double pm)
 {
   int i;
   double delta;


   for (i = 0; i < nb_param; i++)
     if (Rrandom(0.0,1.0) < pm)
        {
          delta = BGX_delta(k,discrete);

          ind -> val_param[i] += delta * (param[i].pmax - param[i].pmin);

          if (ind -> val_param[i] > param[i].pmax)
             ind -> val_param[i] = param[i].pmax;

          if (ind -> val_param[i] < param[i].pmin)
             ind -> val_param[i] = param[i].pmin;
        }
 }

/* ---------------------- fonction Mute_population --------------------- */

// applique la mutation a tous les individus de la population composante
// par composante avec une probabilite pm                                     

void Mute_population (INDIVIDU pop[],int nb_ind, TYPE_PARAM param[],
		 int nb_param,int k, BOOLEEN discrete, double pm, double pm_cross)

 {
   int i;

   for (i = 0; i < nb_ind; i++)
        Mute_individu(&pop[i],param,nb_param,k,discrete,pm);

   // mutation des genes lies au croisement */

   for (i = 0; i<nb_ind; i++)
      if (Rrandom(0.0,1.0)< pm_cross)
        pop[i].type_cross = Erandom(1,3);
 }


/* ------------------------ fonction Selection ------------------------- */

void Selection (INDIVIDU pop[], int nb_ind, int nb_crit, int nb_cont)
 {

   int i,j,k;

   for (i=0; i<nb_ind; i++)
     {
       j = Erandom(0,nb_ind-1);

       do
         k=Erandom(0,nb_ind-1);
       while (k==j);

       if (Dominance_individu(&pop[j],&pop[k],nb_crit,nb_cont))
          pop[nb_ind+i] = pop[j];
       else if (Dominance_individu(&pop[k],&pop[j],nb_crit,nb_cont))
          pop[nb_ind+i] = pop[k];
       else if (pop[j].niche_count > pop[k].niche_count)
              pop[nb_ind+i] = pop[j];
       else
         pop[nb_ind+i] = pop[k];
     }
  }

/* ====== definition des fonctions d'affichage des individu ============ */

/* ---------------- fonction Affiche_individu  ------------------------- */

// Affiche les caractéristiques d'un individu : parametres, contraintes,
// faisabilite (realisable = 1), criteres, front, indice de densite
  
void Affiche_individu (FILE *fichier,INDIVIDU *ind,
					   int nb_param, int nb_crit, int nb_cont)

 {
    int i;

	for (i=0; i<nb_param; i++)
	  fprintf(fichier,"%5e  ",ind->val_param[i]);

	for (i=0; i<nb_cont; i++)
	  fprintf(fichier,"%5e  ",ind->contrainte[i]);

    fprintf(fichier,"%5d  ",ind->realisable);

	for (i=0; i<nb_crit; i++)
	  fprintf(fichier,"%5e  ",ind->critere[i]);

	fprintf(fichier,"%3d  ",ind->front);

   fprintf(fichier,"%5e\n  ",ind->niche_count);

  }

/* ---------------- fonction Affiche_population  ----------------------- */

// Affiche les caractéristiques de toute la population
    
void Affiche_population (FILE *fichier, INDIVIDU pop[], int nb_ind,
						 int nb_param, int nb_crit, int nb_cont)
{
	int i;

	for (i = 0; i < nb_ind; i++)
		Affiche_individu(fichier,&pop[i],nb_param,nb_crit,nb_cont);
}

/* ==================== Programme principal ============================ */

int main()
{
    INDIVIDU pop[NMAX_POP];             // population

    int taille_front[NMAX_POP];         // nombre d'individus par front

    int i,j,nb_front,k,type;

    int nb_ind, nb_param, nb_crit, nb_gene, nb_cont, nb_pop;

    TYPE_PARAM param[NMAX_PARAM];      // contraintes de domaine

    double pm, eta, dec, pm_cross;    

    BOOLEEN discrete;

    FILE *fichier;

    // caracteristiques du probleme d'optimisation 

    nb_param = 2;           // nombre de parametres
    nb_crit = 2;            // nombre de criteres
    nb_cont = 2;            // nombre de contraintes
    
    // contraintes de domaine (bornes des parametres)
    
    param[0].pmin = 1;     
    param[0].pmax  = 10;
    param[1].pmin  = 1;
    param[1].pmax = 10;
    
    // caracteristique generales de l'algorithme genetique 

	nb_gene =200;           // nombre de generation max 
	nb_ind =100;            // nombre d'individus - taille de la population
	nb_pop = 2*nb_ind;      // taille globale avec archive
    
	// parametres des fonctions de sélection - recombinaison - mutation 

    pm = 1.0/nb_param;      // taux de mutation - defaut: 1.0/nb_param
    pm_cross = 0.05;        // taux de mutation du X-gene - defaut: 5%
    
    type= 4;                // type de croisement - defaut : 4
                            // 1 = BGX
                            // 2 = SBX
                            // 3 = BLX
                            // autre = autoadaptatif  
	
    dec = 0.5;              // parametre du BLX - defaut: 0.5                    
    eta = 1.0;              // parametre du SBX - defaut: 1.0
    k = 16;                 // parametre du BGX - defaut: 16
	discrete = VRAI;        // parametre de mutation - defaut VRAI   
   
    // Initialisation des variables globales
    
    cross_bgx = cross_blx = cross_sbx = 0;
    nb_eval = nb_real = 0;
    
    // Initialisation du generateur de nombres aleatoires

	Randomize();
  
   // Corps du NSGA-II
   
   i = 0;
   
   printf("gene = %d\n",i);    //  affichage des generations a l'ecran

   Initialise_population(pop,param,nb_param,nb_crit,nb_pop,nb_cont);

   for (i = 1; i < nb_gene; i++)
     {
       printf("gene = %d\n",i);     //  affichage des generations a l'ecran
       Calcule_dominance (pop,nb_pop,nb_crit,nb_cont,
                          &nb_front,taille_front);

       Affecte_niche (pop,nb_pop,nb_crit,nb_cont,
                      nb_front,taille_front);

       Trie_niche (pop,nb_front,taille_front);

       Selection (pop,nb_ind,nb_crit,nb_cont);

       Croisement (&pop[nb_ind],nb_ind,param,
                   nb_param,k,discrete,eta,dec,type);


       Mute_population(&pop[nb_ind],nb_ind,param,nb_param,
                       k,discrete,pm,pm_cross);
    
       for ( j = 0; j < nb_ind; j++)
          Calcule_criteres (&pop[nb_ind+j],nb_crit,nb_param,nb_cont);       
     }
  
   Calcule_dominance (pop,nb_pop,nb_crit,nb_cont,&nb_front, taille_front);
   Affecte_niche (pop,nb_pop,nb_crit,nb_cont,nb_front,taille_front);
   Trie_niche (pop,nb_front,taille_front);

   // Ouverture du fichier RESGEN.RES pour l'enregistrement des resultats
    
   fichier = fopen("RESGEN.RES","w");
   
   Affiche_population (fichier,pop,nb_ind,nb_param,nb_crit,nb_cont);

   // fermeture du fichier de resultats
   
   fclose(fichier);
  
  // Affichage d'indicateur 
  
  printf("Realisable = %.2f\n",(double)nb_real/(double)nb_eval);
 
  printf("BGX = %.2f\n", (double)cross_bgx / 
                         (double)(cross_bgx+cross_blx+cross_sbx));
  
  printf("BLX = %.2f\n", (double)cross_sbx / 
                         (double)(cross_bgx+cross_blx+cross_sbx));
  
  printf("SBX = %.2f\n", (double)cross_blx / 
                         (double)(cross_bgx+cross_blx+cross_sbx));  
  getchar();
  
return 1;
}


