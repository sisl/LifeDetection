using Pkg
Pkg.activate("experiments")#("..")  # go one level up from experiments/ to LifeDetection/ repo root

using LifeDetection
using SARSOP
using POMDPs

pomdp = LifeDetectionPOMDP(;sample_volume=100)

solver = SARSOPSolver(verbose=true, timeout=200)

policy = load_policy(pomdp, "policy.out")
# policy = solve(solver, pomdp)

plot_alpha_action_heatmap(policy, 100)

rewards, accuracy = simulate_policyVLD(pomdp, policy;type="SARSOP") # Can choose action type to be from SARSOP policy, GREEDY, or CONOPS
