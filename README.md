# GoST — Golang Subject To

Solveur de programmation linéaire en nombres entiers mixtes (MILP), écrit en Go pur, sans dépendance externe.

```go
m := gost.NewModel("production")
a := m.AddVar(0, gost.Inf, 0, gost.Integer, "chairs")
b := m.AddVar(0, gost.Inf, 0, gost.Integer, "tables")
m.AddConstr(gost.NewExpr().Add(2, a).Add(5, b), gost.LessEq, 40, "machine")
m.AddConstr(gost.NewExpr().Add(3, a).Add(2, b), gost.LessEq, 36, "labour")
m.SetObjective(gost.NewExpr().Add(30, a).Add(50, b), gost.Maximize)

res, _ := m.Solve(gost.DefaultParams())
fmt.Println(res.Status, res.Objective, res.Value(a), res.Value(b)) // optimal 470 9 4
```

Lecture d'un fichier MPS (fixe ou libre, `.gz` accepté) :

```go
m, err := mps.ReadFile("model.mps.gz")
p := gost.DefaultParams()
p.TimeLimit = 60 * time.Second
p.Log = os.Stdout          // journal façon CPLEX
res, _ := m.Solve(p)
```

## Génération de colonnes et de contraintes

Trois callbacks, passés dans `Params`, permettent de modifier le modèle pendant la résolution. Les variables et contraintes ajoutées sont ajoutées au `Model` et restent accessibles après `Solve`.

| Callback | Appelé avec | Usage typique |
|---|---|---|
| `LazyConstraints func(*ConstrContext)` | chaque solution candidate entière (et chaque optimum LP pour un modèle continu) | génération de lignes : sous-tours du TSP, contraintes trop nombreuses pour être écrites à l'avance |
| `UserCuts func(*ConstrContext)` | les solutions LP fractionnaires (racine, puis nœuds) | coupes métier valides pour toute solution entière |
| `Pricing func(*PricingContext)` | chaque optimum LP (racine et nœuds) | génération de colonnes, branch-and-price, price-and-branch |

```go
p := gost.DefaultParams()
p.Pricing = func(ctx *gost.PricingContext) {
    col := bestPattern(ctx.Duals())             // sous-problème métier (ex. sac à dos)
    if ctx.ReducedCost(1, col) < -1e-9 {        // coût réduit < 0 en minimisation
        ctx.AddVar(0, gost.Inf, 1, gost.Continuous, "pattern", col)
    }
}
p.LazyConstraints = func(ctx *gost.ConstrContext) {
    if ctx.Value(x)+2*ctx.Value(y) > 12 {
        ctx.AddConstr(gost.NewExpr().Add(1, x).Add(2, y), gost.LessEq, 12, "lazy")
    }
}
```

- `Model.AddVarWithColumn(lo, up, obj, typ, name, col)` permet de modéliser par colonnes (problèmes maîtres).
- Pour un modèle purement continu, `Result.Dual(c)` et `Result.ReducedCost(v)` donnent les duals et coûts réduits (dans le sens du modèle).
- Exemples exécutables dans [`examples/`](examples/README.md) : `cuttingstock` (génération de colonnes), `concurrentflow` (flot concurrent maximum, formulation chemins), `tsp` (sous-tours en lazy + coupes utilisateur), `steinertsp` (TSP orienté sur graphe non complet, nœuds obligatoires, passages multiples), `mipstart` (solution de départ), `lazy`, `production`, `mpsfile`. Tests correspondants : `callbacks_test.go`.
- Règles : quand un callback est fourni, le presolve est désactivé (indices identiques au modèle, pas de réductions duales invalides). Avec `Pricing`, les coupes internes, le strong branching et la propagation sont aussi désactivés ; le LP n'élague qu'après convergence du pricing ; un nœud dont le LP est infaisable est élagué (prévoir des colonnes artificielles dans le maître) ; les variables générées doivent admettre la valeur 0 (`lo <= 0 <= up`) ; une colonne identique (bornes, coût, type, coefficients) à une colonne déjà créée par `AddVarWithColumn` ou `ctx.AddVar` n'est pas ajoutée : `ctx.AddVar` renvoie la variable existante (sinon une branche « z ≤ k » serait annulée par la copie et le branch-and-price pourrait boucler sans fin). Avec un branchement sur variables, le pricer doit malgré tout tenir compte des branchements pour un branch-and-price exact.
- **Reprise à chaud** : après l'ajout de lignes (contraintes paresseuses, coupes), les nouvelles variables d'écart entrent en base et le LP repart de la base courante par le simplexe dual ; après l'ajout de colonnes (pricing), la base reste primal réalisable et le LP repart par le simplexe primal (sans phase 1). Vérifié par `TestWarmStartDynamic` : 4 à 6 itérations en moyenne par re-résolution, contre 90 à 110 pour résoudre le LP final depuis le début.

## Solution de départ (MIP start)

Une solution réalisable (pas forcément optimale), complète ou partielle, peut être donnée au solveur :

```go
m.SetStart(y[3], 1)          // une valeur à la fois (les autres restent libres)
m.SetStartValues(x0)         // ou un vecteur complet (NaN = non donnée)
r, _ := m.Solve(gost.DefaultParams())
fmt.Println(r.StartStatus, r.StartInfo) // accepted / repaired / rejected + détail
```

- **Solution complète et réalisable** (bornes, intégralité, contraintes, et contraintes paresseuses s'il y en a) : elle devient la première solution incumbente. L'élagage par la borne, la fixation par coûts réduits et la propagation de l'objectif sont actifs dès la racine, et RINS part de cette solution.
- **Solution partielle ou irréalisable** : elle est réparée en deux temps. D'abord, les variables entières données sont fixées à leur valeur arrondie et le reste est résolu par un sous-MIP limité à 1 000 nœuds. Si c'est impossible, on cherche la solution réalisable la plus proche (minimum de changements sur les entiers donnés). La solution obtenue devient la première incumbente.
- Après le presolve, la solution est projetée dans le problème réduit ; si une réduction la rend non représentable, elle est gardée de côté et sert de seuil d'élagage (elle est rendue si rien de meilleur n'est trouvé).
- Le départ n'influe jamais sur l'optimalité du résultat. Exemples sur MIPLIB avec l'optimum donné en départ : p2756 110 → 5 nœuds, 10teams 10,8 s → 0,15 s, mod008 25 → 9 nœuds, p0282 49 → 21 nœuds ; sur misc03 l'arbre est au contraire plus grand (693 → 1 057 nœuds, variance des choix de branchement).
- `Model.ReadStart(texte)` lit des lignes « nom valeur » (ou une valeur par ligne dans l'ordre des variables) ; `Model.WriteSolution(r)` écrit une solution dans ce format. En ligne de commande : `-start fichier` et `-writesol fichier`.
- Exemple : [`examples/mipstart`](examples/mipstart/main.go) (localisation d'entrepôts : départ glouton complet, départ partiel, départ irréalisable). Tests : `start_test.go`.

D'autres programmes d'exemple sont dans [`examples/`](examples/README.md) (`go run ./examples/tsp`, etc.).

Outil en ligne de commande (pour les tests et benchmarks) :

```
go run ./cmd/gost -time 60s testdata/p0201.mps.gz
```

## Architecture

| Package | Rôle |
|---|---|
| `gost` (racine) | API de modélisation (`Model`, `Var`, `Expr`, `Params`, `Result`) |
| `mps` | Lecture/écriture MPS (fixe/libre, INTORG, RANGES, BOUNDS UP/LO/FX/FR/MI/PL/BV/LI/UI, OBJSENSE, constante d'objectif) |
| `internal/lu` | Factorisation LU creuse de la base : Markowitz à droite (sous-matrice active par colonnes avec valeurs, recherche limitée Suhl–Suhl sur colonnes et lignes, pivot à seuil, noyau dense Markowitz au-delà de 15 % de densité, tampons réutilisés) ; Gilbert–Peierls conservé en secours (`GOST_MARKOWITZ=0`). Mises à jour **Forrest–Tomlin** (etas produit en secours, `GOST_FT=0`), FTRAN/BTRAN exploitant la creusité et renvoyant les non-zéros, refactorisation adaptative selon le remplissage, réparation des bases singulières |
| `internal/simplex` | Simplexe révisé borné : dual (steepest edge, BFRT/long step, Harris, perturbation des coûts non basiques avec escalade en cas de stagnation, décalage des coûts) avec **phase 1 duale** (problème auxiliaire boxé), primal (phase 1 composite, Devex) pour le nettoyage, **polissage final** à tolérance 1e-9. Mise à l'échelle géométrique + équilibrage, warm start, **ajout de colonnes et de lignes à chaud**, bases extensibles, lignes du tableau |
| `internal/presolve` | Lignes vides/singletons/redondantes/forcing, colonnes fixées/vides, resserrement de bornes, resserrement de coefficients (Savelsbergh), fixation duale, équations à deux variables (agrégation), colonnes singletons implicitement libres et singletons d'écart ; postsolve primal |
| `internal/mip` | Branch-and-cut : propagation de bornes et de l'objectif, branchement par fiabilité (pseudo-coûts + strong branching plafonné), meilleur-d'abord avec plongées, coupes Gomory mixtes, c-MIR avec agrégation et bornes variables, couvertures minimales relevées (sélection par efficacité et parallélisme), redémarrage à la racine, heuristiques (arrondi, plongées guidée/coefficients/fractionnaire, RENS, RINS), fixation par coûts réduits, détection d'objectif entier |

## Validation

- `go test ./...` : LU, relaxations LP (34 instances Netlib/MIPLIB, comparées à HiGHS), ajout incrémental de colonnes/lignes comparé à une résolution complète (400 LP aléatoires), 1 500 MIP aléatoires vérifiés par énumération exhaustive, MIP mixtes aléatoires, aller-retour MPS, génération de colonnes (cutting stock vs LP complet énuméré, branch-and-price), TSP en lazy constraints vs programmation dynamique de Held–Karp.
- `GOST_LPBENCH=<dossier> go test -run TestLPBench -v .` : benchmark LP sur des fichiers `.mps.gz`.
- `go test -run TestMIPLIB .` : 23 instances MIPLIB 3 résolues à l'optimum (≈ 20 s au total).
- Outil de débogage : `-debugsol fichier` (CLI) / `Params.DebugSolution` signale toute coupe, réduction ou élagage qui éliminerait une solution connue.

### Simplexe : LP Netlib (sans presolve, 1 thread, même machine, meilleur de 3)

| Instance | v0.1 | v0.2 | v0.3 | v0.6 | HiGHS 1.16 |
|---|---|---|---|---|---|
| 25fv47 | 0,60 s | 0,32 s | 0,24 s | 0,24 s | 0,15 s |
| 80bau3b | 1,03 s | 0,33 s | 0,25 s | 0,27 s | 0,09 s |
| bnl2 | 1,19 s | 0,21 s | 0,14 s | 0,15 s | 0,06 s |
| degen3 | 1,15 s | 0,35 s | 0,31 s | 0,27 s | 0,15 s |
| d2q06c | 4,7 s | 2,2 s | 1,6 s | 1,4 s | 0,86 s |
| fit2p | 3,6 s | 2,5 s | 1,4 s | 1,4 s | 0,78 s |
| greenbea | 2,2 s | 1,9 s | 1,8 s | 1,3 s | 0,55 s |
| pilot | 8,8 s | 3,5–5 s | 1,9 s | 1,4 s | 0,95 s |
| pilot87 | 22,6 s | 15–25 s | 9,8 s | **4,4 s** | 4,0 s |
| dfl001 | 28,9 s | 17,6 s | 11,1 s | 10,4 s | 8,6 s |
| **Total 22 instances** | **76 s** | **57 s** | **29 s** | **22 s** | **17 s** |

Objectifs identiques à HiGHS (écart relatif max 6e-8, sur pilot87).

v0.6 : factorisation LU réécrite — matrice active stockée par colonnes avec valeurs (plus de recherches dans les lignes), recherche Markowitz sur colonnes **et** lignes (Suhl–Suhl complet), retrait paresseux des lignes pivots (valeurs d'origine pour les lignes jamais modifiées), **noyau dense** avec choix de Markowitz quand la sous-matrice active dépasse 15 % de densité ; FTRAN en espace des étapes, inverses des pivots, L^T indexé par lignes. Sur pilot87, factorisation 25 ms → 8 ms avec un tiers de remplissage en moins (L+U 87k → 58k). Les temps MIP varient de ±50 % selon les instances, car les arbres changent avec les pivots (misc07 repasse juste au-delà de 20 s, qnet1 1,3 s).

v0.3 : mises à jour **Forrest–Tomlin** (au lieu des etas produit), nettoyage final par primal phase 2 sans bascules de bornes destructrices, perturbation amortie et détection de stagnation par rapport au meilleur objectif, cache des évaluations de colonnes dans Markowitz, listes de non-zéros issues de FTRAN, matrice contiguë pour la ligne pivot, tableau d'infaisabilités pour la tarification duale, vérification d'optimalité sans refactorisation quand peu de mises à jour, réutilisation de la factorisation quand la base d'un nœud est déjà installée.

### MIP (1 thread, limite 20 s)

| Instance | GoST v0.3 | GoST v0.4 | GoST v0.5 | HiGHS 1.16 |
|---|---|---|---|---|
| mod008 | 0,8 s (12 333 nœuds) | 0,15 s (13 nœuds) | 0,16 s | 0,8 s |
| gesa2 | 3,7 s | 0,25 s | 0,30 s | 0,6 s |
| fiber | 3,7 s | 1,2 s | **0,43 s** | 0,9 s |
| qnet1 | 4,1 s | 1,9 s | **1,7 s** | 0,6 s |
| bell3a | 9 s | 4,6 s | 4,7 s | 0,2 s |
| vpm2 | limite | ≈ 12 s | **9,6 s** | 1,5 s |
| misc07 | limite (écart 4 %) | limite (écart 3,6 %) | **optimal 19,8 s** | optimal 18 s |
| 10teams | 5,6 s | 5,1 s | 5,0 s | 3,3 s |
| fixnet6 | 11 s | 16 s | 22 s | 2,6 s |
| autres instances faciles (p0033, p0201, p0282, p0548, p2756, lseu, egout, flugpl, gt2, khb05250, misc03, stein27, vpm1, rgn, enigma, bell5, dcmulti, blend2, air03) | < 1,3 s | < 1 s | < 1 s | < 2,6 s |
| mas76 | limite (écart 0,7 %) | limite (écart 0,7 %) | limite (écart 0,7 %) | limite (écart 1,9 %) |
| pk1 | limite, sol. 13 | limite, sol. 14 | limite, sol. 14 | limite, sol. 15 |

v0.5 : presolve par **élimination de variables** (équations à deux variables, colonnes singletons implicitement libres, singletons d'écart à coût nul), avec postsolve primal ; **résolution déterministe** sans limite de temps (sous-MIP limités en nœuds, effort des coupes compté, sommes indépendantes de l'ordre d'itération des maps) — vérifié par `TestDeterminism`. La durée de fixnet6 varie au gré de l'arbre (16 à 22 s) : ses coupes restent trop faibles (borne racine 3 287 contre 3 971 pour HiGHS).

v0.4 : coupes **c-MIR** (Marchand–Wolsey, agrégation de lignes, substitution par bornes simples et bornes variables y ≤ u·x, complémentation), **couvertures minimales relevées** (relèvement séquentiel exact par programmation dynamique), limite de densité des coupes adaptée aux lignes denses, **redémarrage** à la racine quand ≥ 20 % des entiers sont fixés (coupes conservées, cutoff tenant compte de l'écart toléré), **propagation de l'objectif** (cutoff), fixation par coûts réduits aux nœuds dès qu'un cutoff existe.

Gap fermé à la racine (relaxation après coupes, avant branchement) : gesa2 81 → 95 %, qnet1 23 → 65–72 %, rgn 32 → 99 %, vpm2 45 → 66–71 %, mod008 0 → 63 %, p0033 43 → 93 %, vpm1 63 → 100 %.

## Feuille de route vers un niveau « commercial »

0. **MIP** : analyse de conflits, coupes de flot plus fortes (fixnet6), probing et cliques, pool de coupes réutilisé dans l'arbre, colonnes dominées/parallèles et lignes parallèles au presolve (écart restant avec HiGHS sur bell3a et fixnet6).
1. **Simplexe** : matrice ligne partitionnée (colonnes hors base) pour la ligne pivot, phase 1 duale plus courte (greenbea : 5 500 itérations contre 2 800 pour HiGHS), FTRAN/BTRAN hypercreux pour les problèmes très creux, tarification duale partielle/parallèle, crash basis, méthode de barrière (points intérieurs) + crossover.
2. **Presolve** : agrégation de doubletons, colonnes/lignes dominées et dupliquées, substitution de variables implicitement libres, détection de variables implicitement entières, sondage (probing) et graphe de conflits.
3. **Coupes** : MIR/flow cover, lifted knapsack cover complet, clique, zéro-demi, implied bounds, pool de coupes global, coupes locales dans l'arbre.
4. **Recherche** : parallélisme (arbre partagé entre goroutines, déterminisme optionnel), redémarrages, analyse de conflits, symétries (orbitopes).
5. **Heuristiques** : feasibility pump, local branching, solution polishing, ordonnancement adaptatif des heuristiques.
6. **API** : contraintes indicatrices et SOS1/SOS2, semi-continues, callback de solution incumbente, pricing de Farkas (nœuds infaisables en branch-and-price), branchement personnalisé (Ryan–Foster), lecteur format LP.

## Développement

- `GOST_LPDEBUG=1` : traces du simplexe ; `GOST_LPSTATS=1` : statistiques LP (phase 1, dual, primal, polissage) ; `GOST_ETAGROWTH`, `GOST_MARKOWITZ`, `GOST_FT`, `GOST_MKSEARCH`, `GOST_ROWSEARCH`, `GOST_DENSE` (seuil du noyau dense), `GOST_REFREQ`, `GOST_VERIFYUPD` : réglages expérimentaux de la factorisation et du simplexe ; `GOST_NOMIR`, `GOST_CMIR_AGGR`, `GOST_CMIR_SLACK`, `GOST_CMIR_CUTS` : réglages des coupes c-MIR ; `GOST_TREESTATS=1` : statistiques de l'arbre ; `GOST_DUMPBASIS=fichier` : écrit la 50ᵉ base factorisée (banc d'essai `GOST_BASES=dossier go test -run TestFactorSpeed ./internal/lu`) ; `GOST_DUMPRESTART=fichier.mps` : écrit le problème redémarré ; `GOST_NOSUBMIP`, `GOST_NODIVE`, `GOST_NORESTART` : désactivent des heuristiques ou le redémarrage ; `GOST_STARTCUTOFF=1` : force l'usage de la solution de départ comme simple seuil d'élagage (test) ; `GOST_PRIMALWARM=0` : reprise après ajout de colonnes par phase 1 duale (ancien comportement) ; `GOST_PRESOLVE_OFF=aggr` désactive les éliminations de variables ; `GOST_HEURDEBUG=1` : traces des heuristiques ; `GOST_PRESOLVE_OFF=coef,dual,forcing,bound` : désactive des réductions.
- `GOST_PROF=<instance> go test -run TestProfileInstance -cpuprofile cpu.out .` pour profiler.
- Le chemin de module est `gost` : renommez-le (ex. `github.com/<org>/gost`) dans `go.mod` avant publication.
