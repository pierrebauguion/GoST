# Exemples GoST

Chaque dossier est un programme autonome, à lancer depuis la racine du dépôt :

| Exemple | Commande | Ce qu'il montre |
|---|---|---|
| `production` | `go run ./examples/production` | Modèle en nombres entiers construit avec l'API (`AddVar`, `AddConstr`, `SetObjective`) |
| `mpsfile` | `go run ./examples/mpsfile [fichier.mps.gz] [limite]` | Lecture d'un fichier MPS et journal du solveur |
| `lazy` | `go run ./examples/lazy` | Contrainte paresseuse (`Params.LazyConstraints`) ajoutée seulement quand une solution candidate la viole |
| `cuttingstock` | `go run ./examples/cuttingstock` | Génération de colonnes (`Params.Pricing`, `AddVarWithColumn`, duals) : relaxation LP puis solution entière en price-and-branch |
| `concurrentflow` | `go run ./examples/concurrentflow` | Flot concurrent maximum en formulation chemins : génération de colonnes par plus courts chemins sur les duals des capacités, décomposition du flot par chemin, vérification par la formulation arcs |
| `tsp` | `go run ./examples/tsp [n]` | Voyageur de commerce : sous-tours en contraintes paresseuses + coupes utilisateur (`Params.UserCuts`) |

Les tests automatiques correspondants sont dans `callbacks_test.go` à la racine (`TestColumnGenerationCuttingStock`, `TestLazySubtourTSP`, `TestLazyTSPLarger`, `TestLPDualsAndRowGeneration`).
