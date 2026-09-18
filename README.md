# LongGuide

LongGuide is a centralized system design for memory-bounded, long-context
LLM fine-tuning. It uses a small **guide set** for backpropagation (BP) and
uses the remaining **bulk set** for zeroth-order optimization (ZOO), while
keeping the entire workload in one training job.

The source tree contains the existing training runtime and experiment entry
points; this README describes the LongGuide interpretation and centralized
workflow. Existing source files, configurations, and experiment entry points
are retained as-is.

## Centralized workflow

LongGuide treats all examples as part of one long-context training workload:

1. A low-cost scan partitions examples into length, structure, difficulty, and
   sparsity strata.
2. A small candidate pool receives probe gradients. A coverage-aware selector
   chooses guide examples by marginal gradient-space novelty per BP cost.
3. The selected guide set runs real BP. Layer-local gradient sketches are
   streamed into a compact guidance basis.
4. Bulk examples use antithetic ZOO perturbations sampled from the guidance
   basis plus a residual random component.
5. Audits monitor held-out loss, BP/ZOO alignment, projection residual, ZOO
   variance, basis drift, and execution cost before a guidance snapshot is
   reused or refreshed.

There is no client/server split, federated aggregation, or cross-device data
ownership assumption in this workflow. Multiple GPUs may still be used by the
underlying training runtime; they remain workers of the same centralized job.

## Design principles

- **Coverage per BP cost:** guide selection optimizes gradient-space coverage,
  rather than selecting only the longest or highest-loss examples.
- **Immutable guidance snapshots:** a snapshot binds a model state, guide-set
  digest, layer-local basis, rank, precision policy, mask policy, and quality
  statistics. A ZOO positive/negative pair uses one snapshot.
- **Residual exploration:** ZOO retains a random component so uncovered bulk
  directions are not permanently discarded.
- **Pair-consistent sparsity:** when a sparse long-context backend is used, the
  positive and negative perturbations share the same token mask and execution
  state.
- **Quality before reuse:** refresh and rollback decisions are based on audit
  signals and their measured cost, rather than a fixed refresh interval.

## Installation

The inherited experiments use the original Python 3.7 environment:

```bash
conda create --name LongGuide python=3.7.15
conda activate LongGuide
pip install -r requirements.txt
conda install mpi4py=3.0.3=py37hf046da1_1
conda install six==1.15.0

cd FedML
git submodule init
git submodule update
cd ..
```

Some data preparation commands in the inherited tree download public datasets;
see the corresponding task README files for their original paths and formats.

## Repository layout

- `FedML/`: the inherited training runtime and model utilities.
- `data/`, `data_manager/`, `data_preprocessing/`: data preparation and
  feature-loading utilities.
- `model/`: model definitions.
- `training/` and `experiments/`: training and experiment entry points.
- `requirements.txt`: Python dependencies.

The centralized LongGuide workflow is a system-level organization of the
existing training components. This repository does not claim that every legacy
entry point has been converted to that workflow.

## Provenance

Only this README and the project-level documentation are LongGuide-specific;
the implementation tree is retained without source-code rewrites.
