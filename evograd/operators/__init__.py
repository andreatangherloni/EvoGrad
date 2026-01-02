"""
EvoGrad operators module.

This module provides the building blocks for evolutionary algorithms:

Sampling (population initialisation):
    - UniformSampling: Uniform random sampling
    - LatinHypercubeSampling: Better space coverage
    - NormalSampling: Gaussian around center
    - LogUniformSampling: Log-scale sampling
    - HaltonSampling: Quasi-random low-discrepancy

Selection (parent selection):
    - TournamentSelection: Tournament-based selection
    - RouletteSelection: Fitness-proportionate selection
    - RankSelection: Rank-based selection
    - RandomSelection: Uniform random selection
    - TruncationSelection: Select from top fraction (samples within elite set)
    - TopKSelection: Deterministic top-k WITHOUT replacement
    - StochasticUniversalSampling: Evenly-spaced roulette

Crossover (recombination):
    - SBXCrossover: Simulated Binary Crossover (GA)
    - BlendCrossover: BLX-alpha crossover (GA)
    - BinomialCrossover: DE-style binomial crossover
    - ExponentialCrossover: DE-style exponential crossover
    - UniformCrossover: Simple uniform crossover
    - ArithmeticCrossover: Weighted average of parents
    - NPointCrossover: N-point crossover

Mutation:
    - PolynomialMutation: Bounded polynomial mutation (GA)
    - GaussianMutation: Gaussian/normal perturbation
    - UniformMutation: Uniform random reset
    - NonUniformMutation: Decreasing perturbation over time
    - BoundaryMutation: Reset to boundary values
    - NoMutation: Identity (no mutation)
    - CombinedMutation: Chain multiple mutations

Repair (bounds handling):
    - ClipRepair: Clamp to bounds
    - ReflectRepair: Bounce off boundaries
    - WrapRepair: Periodic wrapping
    - RandomRepair: Reset violating genes
    - BoundsRepair: Configurable repair method
    - SoftClipRepair: Smooth differentiable clip
    - PenaltyRepair: Compute penalty instead of repair
    - NoRepair: Identity (no repair)

Survival (generational replacement):
    - MergeSurvival: (μ+λ) Select from parents + offspring
    - CommaSurvival: (μ,λ) Select only from offspring
    - ReplaceWorstSurvival: Steady-state replacement
    - FitnessSurvival: Simple fitness-based truncation
    - AgeSurvival: Age-based with fitness tie-breaking
"""

from evograd.operators.sampling import (
    Sampling,
    UniformSampling,
    LatinHypercubeSampling,
    NormalSampling,
    LogUniformSampling,
    HaltonSampling,
)

from evograd.operators.selection import (
    Selection,
    TournamentSelection,
    RouletteSelection,
    RankSelection,
    RandomSelection,
    TopKSelection,
    TopKSelection,
    TruncationSelection,
    StochasticUniversalSampling,
)

from evograd.operators.crossover import (
    Crossover,
    SBXCrossover,
    BlendCrossover,
    BinomialCrossover,
    ExponentialCrossover,
    UniformCrossover,
    ArithmeticCrossover,
    NPointCrossover,
)

from evograd.operators.mutation import (
    Mutation,
    PolynomialMutation,
    GaussianMutation,
    UniformMutation,
    NonUniformMutation,
    BoundaryMutation,
    NoMutation,
    CombinedMutation,
)

from evograd.operators.repair import (
    Repair,
    ClipRepair,
    ReflectRepair,
    WrapRepair,
    RandomRepair,
    BoundsRepair,
    SoftClipRepair,
    PenaltyRepair,
    NoRepair,
    RepairMethod,
)

from evograd.operators.survival import (
    Survival,
    MergeSurvival,
    CommaSurvival,
    ReplaceWorstSurvival,
    FitnessSurvival,
    AgeSurvival,
    get_survival,
)

__all__ = [
    # Sampling
    "Sampling",
    "UniformSampling",
    "LatinHypercubeSampling",
    "NormalSampling",
    "LogUniformSampling",
    "HaltonSampling",
    # Selection
    "Selection",
    "TournamentSelection",
    "RouletteSelection",
    "RankSelection",
    "RandomSelection",
    "TopKSelection",
    "TruncationSelection",
    "StochasticUniversalSampling",
    # Crossover
    "Crossover",
    "SBXCrossover",
    "BlendCrossover",
    "BinomialCrossover",
    "ExponentialCrossover",
    "UniformCrossover",
    "ArithmeticCrossover",
    "NPointCrossover",
    # Mutation
    "Mutation",
    "PolynomialMutation",
    "GaussianMutation",
    "UniformMutation",
    "NonUniformMutation",
    "BoundaryMutation",
    "NoMutation",
    "CombinedMutation",
    # Repair
    "Repair",
    "ClipRepair",
    "ReflectRepair",
    "WrapRepair",
    "RandomRepair",
    "BoundsRepair",
    "SoftClipRepair",
    "PenaltyRepair",
    "NoRepair",
    "RepairMethod",
    # Survival
    "Survival",
    "MergeSurvival",
    "CommaSurvival",
    "ReplaceWorstSurvival",
    "FitnessSurvival",
    "AgeSurvival",
    "get_survival",
]
