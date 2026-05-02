# A Graph-Based Framework for Extending Metric Differential Privacy Mechanisms

MATLAB implementation accompanying the paper:  
**"A Graph-Based Framework for Extending Metric Differential Privacy Mechanisms"**

---

## Overview

This codebase implements and evaluates an **extension-based mDP framework** for large-scale geographic domains.  
The core idea: instead of solving a full perturbation-matrix optimization over all N locations (computationally prohibitive), the algorithm first optimizes over a coarse anchor grid (~132 points), then recursively extends the mechanism to finer grids through local interpolation operations.

Three tree-based extension variants are implemented (**MLaEt-A**, **MLaEt-M**, **MLaEt-O**), alongside all baselines reported in the paper.

---

## Repository Structure

```
.
├── parameters.m                      # Global parameters (epsilon, grid size, NR_TEST, etc.)
│
├── — Main Experiments (Proposed Methods) —
├── main_tree_interpolation.m         # MLaEt-A: log-convex interpolation extension (Table 2 main)
├── main_tree_MWE.m                   # MLaEt-M: McShane-Whitney extension
├── main_tree_LP.m                    # MLaEt-O: LP-based cell-wise extension
│
├── — Main Experiments (Baselines) —
├── main_2norm.m                      # EM + AIPO baselines (ℓ₂ norm, main results)
├── EM_EMBR.m                         # EM and EM+BR (Bayesian remapping) baselines
├── main_laplace.m                    # Planar Laplace mechanism baseline
├── main_LP_A.m                       # LP-A (direct LP on coarse grid) baseline
│
├── — Appendix Experiments —
├── main_granularity_appendix.m       # Grid granularity sensitivity analysis
├── main_2norm_ori.m                  # ℓ₂ results with fixed random seed
│
├── — Single-Step Extension (Intermediate) —
├── main_extension_interpolation.m    # Single extension round, interpolation variant
├── main_extension_MWE.m              # Single extension round, McShane-Whitney
│
├── — Core Algorithm Modules —
├── interpolation_extension_k.m       # Workspace script: refines grid k-fold, returns allPoints + W
├── MWE_cell_BFS.m                    # McShane-Whitney extension, cell-wise BFS (used in tree methods)
├── mcshane_whitney_extension.m       # McShane-Whitney extension, global version
├── lp_extension_cellwise.m           # LP-based cell-wise extension operator
├── get_child_cells.m                 # Cell refinement: subdivides parent cells into k×k children
│
├── save_point_data.m                 # Export obfuscated location data for visualization
│
├── functions/                        # Shared utility functions
│   ├── uniform_anchor.m              # Build anchor grid via uniform cell partitioning
│   ├── log_interp_cell_BFS.m         # Log-convex interpolation (cell-wise BFS, main extension op)
│   ├── logconv_interp.m              # Log-convex interpolation (global version)
│   ├── perturbation_cal_apo.m        # AIPO seed mechanism (LP on anchor grid)
│   ├── partition_grid.m              # Partition domain into coarse grid for LP-A
│   ├── compute_mDP_violation.m       # Empirical mDP violation rate computation
│   ├── distance_matrix.m             # Pairwise ℓ_p distance matrix
│   ├── lonlat_to_xy.m                # Project (lon, lat) to local XY coordinates (km)
│   ├── filter_coords_by_range.m      # Filter nodes within a bounding box
│   ├── convert_grid_to_fine_perturbation.m  # Map coarse LP solution back to full domain
│   ├── benchmarks/
│   │   ├── perturbation_cal_em.m     # Exponential mechanism
│   │   ├── perturbation_cal_laplace.m # Planar Laplace mechanism
│   │   ├── perturbation_cal_rmp.m    # Bayesian remapping (EM+BR)
│   │   ├── perturbation_cal_lp.m     # LP solver for LP-A baseline
│   │   ├── perturbation_cal_copt.m   # COPT baseline
│   │   └── perturbation_cal_tem.m    # TEM baseline
│   └── haversine/                    # Haversine distance utilities
│
├── datasets/                         # Road-network node data
│   ├── rome/nodes.mat
│   ├── nyc/nodes.mat
│   └── london/nodes.mat
├── results/                          # Saved experiment outputs
└── intermediate/                     # Precomputed data (graph shortest paths, etc.)
```

---

## Algorithm Overview

### Tree-Based Extension (Proposed)

1. **Anchor grid construction** (`uniform_anchor`): partition the city map into a uniform grid (~132 anchor cells, ~4.77 km cell size for Rome). Each real location is mapped to its surrounding anchor corners via bilinear weights.

2. **Seed mechanism optimization** (`perturbation_cal_apo`): solve an LP on the coarse anchor grid using the approximated cost matrix `c_approx = W₀ᵀ · L₀`, where `W₀` maps real locations to anchors and `L₀` is the Euclidean loss matrix.

3. **Recursive extension** (`interpolation_extension_k` + extension operator): extend from coarse to fine grids through T=3 rounds. At each round, the grid is densified (k-fold per axis), and the perturbation matrix is extended using one of three operators:
   - **Log-convex** (`log_interp_cell_BFS`): bilinear interpolation in log-probability space — MLaEt-A
   - **McShane-Whitney** (`MWE_cell_BFS`): envelope-based extension — MLaEt-M
   - **LP-based** (`lp_extension_cellwise`): local LP per cell — MLaEt-O

Grid progression for Rome: 132 anchors → 483 → 1,845 → 7,209 points.

### Privacy Budget Split

The total budget ε is split between ε₁ (seed LP) and ε₂ (extension):  
`ε₁² + ε₂² ≤ ε²` (ℓ₂-composition for p=2).  
The optimal split is searched over `NR_EPSILON_INTERVAL` discrete values.

---

## Key Parameters (`parameters.m`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NR_TEST` | 2 | Number of repeated random trials |
| `EPSILON_MAX` | 3 | Number of ε values tested (ε = 0.5, 1.0, 1.5) |
| `NR_PER_LOC` | 20 | Number of obfuscation candidates (K) |
| `cell_size` | [4.77, 6.87, 6.78] | Anchor cell size per city (km): Rome, NYC, London |
| `GRID_SIZE_LP` | 18 | LP-A grid size (18×18) |
| `NR_EPSILON_INTERVAL` | 5 | Budget-split search resolution |
| `SAMPLE_SIZE_PPR` | 1000 | Sample size for mDP violation check |
| `SCALE` | 2 | Coverage scale (2 = full city) |

---

## Output Format

All main scripts print results in the format used by the paper tables:  
**9 values** = 3 epsilon values × 3 extension layers, printed as `mean±std` separated by ` & `.

Example: `10.24±0.76 & 10.34±1.72 & 102.26±0.53 & ...`  
Columns: (ε=0.5,L1) (ε=1.0,L1) (ε=1.5,L1) (ε=0.5,L2) ... (ε=1.5,L3)

Results are also appended to text files in `results/`.

---

## Running Experiments

All scripts are standalone — run from the `tree_artifact/` root directory in MATLAB.  
Change `city_idx` in the city loop to select Rome (1), NYC (2), or London (3).

```matlab
% Example: run main tree-interpolation experiment on Rome
% Open MATLAB, cd to tree_artifact/, then:
run('main_tree_interpolation.m')
```

### Reproducing Paper Results (Table 2, Rome, ℓ₂)

| Method | Script |
|--------|--------|
| EM | `EM_EMBR.m` |
| EM+BR | `EM_EMBR.m` |
| Laplace | `main_laplace.m` |
| LP-A | `main_LP_A.m` |
| AIPO (MLaEt seed only) | `main_2norm.m` |
| **MLaEt-A** | `main_tree_interpolation.m` |
| **MLaEt-M** | `main_tree_MWE.m` |
| **MLaEt-O** | `main_tree_LP.m` |
