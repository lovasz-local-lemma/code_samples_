# Numerics and engineering — modules with tests

| Read | What it shows |
| --- | --- |
| [`fft/fft.cpp`](fft/fft.cpp) / [`fft.hpp`](fft/fft.hpp) | From my signal-processing library: a radix-2 FFT with a generated bit-reversal table, one butterfly shared by compile-time backend policies (serial, OpenMP, standard-C++ fork/join tasks), reusable plans, inverse transforms, and `Plan::measured`, which warms every compiled backend, checks agreement, and picks the lowest median of three timings. |
| [`ordered-set/ordered_set.hpp`](ordered-set/ordered_set.hpp) | An owning left-leaning red-black tree with insertion, deletion, cached subtree sizes and executable invariants (colour, order, black height, size), checked against `std::set` over 7,000 random operations. |
| [`render-graph/render_graph.cpp`](render-graph/render_graph.cpp) / [`.hpp`](render-graph/render_graph.hpp) | Dependency ordering of selected outputs, current-frame cycle diagnosis with the offending path, explicit previous-frame reads, and revision-based plan caching with immutable old plans. |
| [`bilateral-grid/bilateral_grid.cpp`](bilateral-grid/bilateral_grid.cpp) / [`.hpp`](bilateral-grid/bilateral_grid.hpp) | Joint bilateral filtering by splat → separable 3D blur → trilinear slice on a guide/signal grid, with a memory budget, compared against a direct joint bilateral filter (RMSE 5.6e-4 on the test fixture). |
