# Exemples GoST

Chaque dossier est un programme autonome, à lancer depuis la racine du dépôt :

| Exemple | Commande | Ce qu'il montre |
|---|---|---|
| `production` | `go run ./examples/production` | Modèle en nombres entiers construit avec l'API (`AddVar`, `AddConstr`, `SetObjective`) |
| `mpsfile` | `go run ./examples/mpsfile [fichier.mps.gz] [limite]` | Lecture d'un fichier MPS et journal du solveur |
| `mipstart` | `go run ./examples/mipstart` | Solution de départ (MIP start) sur une localisation d'entrepôts avec capacités : départ glouton complet (accepté), partiel (complété), irréalisable (réparé au plus proche) |
| `binpacking` | `go run ./examples/binpacking` | Bin packing (mise en lots de base) : formulation d'affectation, bris de symétrie, borne L1 et heuristique FFD comparées à l'optimum |
| `lotsizing` | `go run ./examples/lotsizing` | Lot sizing multi-produits à capacité (CLSP) : taille des lots de production, inégalités (l,S) de Barany–Van Roy–Wolsey en coupes utilisateur |
| `batchscheduling` | `go run ./examples/batchscheduling` | Machine à traitement par lots (four de déverminage) : formulation naïve vs formulation « par représentant » (sans symétrie), heuristique FFLPT |
| `orderbatching` | `go run ./examples/orderbatching` | Regroupement de commandes en tournées de préparation (entrepôt, politique « return ») : comparaison avec FCFS et l'heuristique « seed » |
| `jobshop` | `go run ./examples/jobshop` | Job shop ft06 de Fisher–Thompson (optimum connu 55) : modèle disjonctif big-M, diagramme de Gantt texte |
| `graphcoloring` | `go run ./examples/graphcoloring [hard]` | Coloration de graphes DIMACS (Mycielski, reines) : heuristique DSatur, contraintes d'arêtes vs de cliques |
| `lazy` | `go run ./examples/lazy` | Contrainte paresseuse (`Params.LazyConstraints`) ajoutée seulement quand une solution candidate la viole |
| `cuttingstock` | `go run ./examples/cuttingstock` | Génération de colonnes (`Params.Pricing`, `AddVarWithColumn`, duals) : relaxation LP puis solution entière en price-and-branch |
| `concurrentflow` | `go run ./examples/concurrentflow` | Flot concurrent maximum en formulation chemins : génération de colonnes par plus courts chemins sur les duals des capacités, décomposition du flot par chemin, vérification par la formulation arcs |
| `steinertsp` | `go run ./examples/steinertsp` | TSP de Steiner sur un graphe orienté non complet : seul un sous-ensemble de nœuds est obligatoire, passages multiples autorisés (variables entières), connexité en contraintes paresseuses + coupes par flot maximum, marche eulérienne, vérification par plus courts chemins + Held–Karp |
| `tsp` | `go run ./examples/tsp [n]` | Voyageur de commerce : sous-tours en contraintes paresseuses + coupes utilisateur (`Params.UserCuts`) |

Les tests automatiques correspondants sont dans `callbacks_test.go` à la racine (`TestColumnGenerationCuttingStock`, `TestLazySubtourTSP`, `TestLazyTSPLarger`, `TestLPDualsAndRowGeneration`).
