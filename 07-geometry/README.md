# Geometry — Fortune's Loom, Apollonius diagrams, automata

## Fortune's Loom — sweepline Voronoi with the beachline visible

Portfolio (live): https://lovasz-local-lemma.github.io/projects/fortune_crystal/index.html

| Read | What it shows |
| --- | --- |
| [`arc_beachline.cpp`](fortune-cpp/arc_beachline.cpp) | One threaded-BST node type driven by five swappable balance policies — AVL, red-black, treap, splay, size-balanced — with a single symmetric `rotate()`, policy-specific insert fix-ups, five deletion strategies (transplant, successor splice, RB delete-fixup with `x_parent` tracking, splay unlink ordering), and a validator that cross-checks in-order traversal against the threaded list. Applied to the Fortune beachline, where each node is a parabolic arc and `prev`/`next` give O(1) neighbour triples. |
| [`beachline.cpp`](fortune-cpp/beachline.cpp) | The classic de Berg leaf-oriented beachline: arcs on leaves, breakpoints on internal nodes, AVL-balanced, extreme-leaf pointers cached per subtree so a breakpoint's two defining sites are O(1) during descent; `split_arc` builds the 3-leaf/2-breakpoint subtree in place. |
| [`fortune_engine.cpp`](fortune-cpp/fortune_engine.cpp) | The sweep: a priority queue with a provably transitive comparator (the file explains why the old epsilon comparator was undefined behaviour), lazy circle-event invalidation, same-y adjacent insertion, and a cocircular-aware circle handler that collapses the maximal run of arcs meeting at one vertex, terminating d−1 edges and opening one replacement edge — degree-d vertices on grids and rings handled structurally, with the pre-fix behaviour kept behind switches for A/B comparison. |
| [`geometry.cpp`](fortune-cpp/geometry.cpp) | The numerical kernel: a predicate kernel with scale-relative tolerance bands and an audit trail, parabola intersection with relative-tolerance handling of the equal-y and near-linear regimes, long-double circumcircle, vertex welding for cocircular clusters, Liang–Barsky clipping, and edge finalisation that orients half-open bisector rays by the third circumcircle site. |
| [`lazy_scalar.cpp`](fortune-cpp/lazy_scalar.cpp) | A formula-preserving scalar: a shared expression DAG over `+ − × ÷ √` with lazily cached interval bounds, a `sign()` that only materialises when the interval straddles zero, forced-comparison provenance counters and label-collapsed printing for the foldable viewer — a small lazy-exact number in the spirit of CGAL's `Lazy_exact_nt`, documented honestly as tolerance band plus tie policy. |
| [`apollonius_engine.cpp`](fortune-cpp/apollonius_engine.cpp) / [`.h`](fortune-cpp/apollonius_engine.h), [`apollonius_dcel.h`](fortune-cpp/apollonius_dcel.h) | Apollonius (additively weighted Voronoi) diagrams: the tangent-circle vertex solve reduced to a 2×2 linear system plus one quadratic, hyperbola-branch bisectors, and a header-only half-edge builder (angular sort, CCW next-linking, cycle walking, bounding-box perimeter ring, lens repair). The diagram is built over all site triples with sampled visibility along pair curves — a direct construction, not a Karavelas–Yvinec incremental algorithm; the code says so. |

## Automata — divisibility as a regular expression

[`automata/automata.cpp`](automata/automata.cpp): the remainder DFA for "binary numbers divisible by n" (state
`r`, bit `b` → `(2r + b) mod n`) eliminated state by state into a regular expression, with the empty language
and ε kept distinct, a degree-based elimination order and expression-size budgets; checked against integer
modulo.
