# LifeDetection

The LifeDetection Suite [1] problem with the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface. 


[1] Kim, G. R., Warner, H., Eddy, D., Astle, E., Booth, Z., Balaban, E., & Kochenderfer, M. J. (2025). Adaptive Science Operations in Deep Space Missions Using Offline Belief State Planning. ([link](https://arxiv.org/abs/2510.08812)).

---

`LifeDetection` is a Julia package for simulating and analyzing a POMDP formulation of the Enceladus Orbilander's proposed Life Detection instrument suite. The `LifeDetectionPOMDP` models a sequential decision-making problem for detecting biotic signals under limited sample volume, noisy instruments, and costly actions. The agent must decide when to accumulate samples, which instruments to deploy, and when to declare a conclusion (biotic or abiotic). Functions to visualize alpha-action heatmaps, and to perform simulation rollouts to evaluate policy performance are included.



![orbilander](orbilander.png)

*[2] MacKenzie, S. M., et al., (2021), The Enceladus Orbilander Mission Concept: Balancing Return and Resources in the Search for Life, Planet. Sci. J, 2(77), doi: 10.3847/PSJ/abe4da.*


## Installation

Use the Julia package manager to install directly from GitHub:

```
julia
using Pkg
Pkg.add(url="https://github.com/sisl/LifeDetection", rev="package-refactor")
```

## States

Each state is represented by a discrete pair:

- Sample volume: the amount of material currently available in the storage container
- Life state:
    - 1 — abiotic
    - 2 — biotic
    - 3 — terminal (after declaration)

## Actions

At each step, the agent may choose one of the following actions:

- Instrument actions (`a = 1:inst-1`)
    - Deploy a scientific instrument to gather observations
    - Each instrument consumes a fixed amount of sample volume (`sample_use[a]`)
    - Instruments produce noisy, probabilistic observations conditioned on the true life state
- Accumulate action (`a = inst`)
    - Increases sample volume by an accumulation rate (`ACC_RATE`)
    - Accumulation may be stochastic, controlled by `std_fraction`
    - Declaration actions:
- `a = inst + 1`: declare abiotic
- `a = inst + 2`: declare biotic
    - Declaration actions immediately transition the system to a terminal state.

## Transition Model

- Instrument actions deterministically reduce the sample volume and leave the life state unchanged.
- Accumulate action increases the sample volume up to a maximum capacity:
    - If `std_fraction > 0`, accumulation is stochastic (Gaussian-distributed increments).
    - The life state after accumulation is resampled according to the Bayesian Network prior.
- Declaration actions transition deterministically to a terminal state.

## Observation Model

- Observations are discrete and instrument-specific.
- For instrument actions:
    - Observations are sampled from conditional probability distributions defined by a Bayesian Network (`ACTION_CPDS`).
    - Some observations originate from continuous sensors and are discretized into bins.
- For accumulate, declaration, or terminal states:
    - A null observation is returned.
The total observation space scales with both the maximum number of observations per instrument (max_obs) and the current sample volume.

## Reward Model

The reward function encodes decision accuracy, efficiency, and feasibility:

- Declaration rewards:
    - Correct declaration → low or zero penalty
    - Incorrect declaration → penalty scaled by:
        - `λ` (decision penalty weight)
        - `τ` (asymmetry for false abiotic declarations)
    - Instrument actions:
        - Penalized proportionally to remaining sample volume to encourage efficiency
    - Infeasible actions (e.g., insufficient sample volume):
        - Large negative reward (`MAX_PENALTY`)

### Key Hyperparameters

| Parameter        | Description                                      |
|------------------|--------------------------------------------------|
| `sample_volume`  | Maximum sample capacity                          |
| `ACC_RATE`       | Mean accumulation rate                           |
| `std_fraction`   | Accumulation stochasticity                       |
| `inst`           | Number of instruments (+ accumulate action)      |
| `sample_use`     | Sample cost per instrument                       |
| `max_obs`        | Number of discrete observation bins              |
| `λ`              | Penalty weight for incorrect declarations        |
| `τ`              | Asymmetry for false abiotic declarations          |
| `discount`       | Discount factor                                  |
| `MAX_PENALTY`    | Penalty for infeasible actions                   |


## Example use
(also included in `experiments/example.jl`)

```
using Pkg
Pkg.activate("experiments") # Activate environment for reproducibility

using LifeDetection
using POMDPs
using SARSOP

# Create a LifeDetection POMDP
pomdp = LifeDetectionPOMDP(; sample_volume=100)

# Solve policy using SARSOP
solver = SARSOPSolver(verbose=true, timeout=200)
# policy = solve(solver, pomdp)
policy = load_policy(pomdp, "policy.out") # or use solve(solver, pomdp) to generate new

# Visualize alpha-action heatmap
plot_alpha_action_heatmap(policy, pomdp, 100)

# Simulate policy and evaluate
rewards, accuracy = simulate_policyVLD(pomdp, policy; type="SARSOP") 
```

## Custom Policy Evaluation

You can simulate policies using different built in action selection types:
```
# Greedy policy simulation
rewards_greedy, accuracy_greedy = simulate_policyVLD(pomdp, policy; type="GREEDY")

# CONOPS simulation
rewards_conops, accuracy_conops = simulate_policyVLD(pomdp, policy; type="CONOPS")
```