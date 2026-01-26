module LifeDetection


using BayesNets
using POMDPs
using POMDPTools
using Plots
using LaTeXStrings
using Printf

include("common/utils.jl")
include("common/plotting.jl")
include("common/bayes_net_helpers.jl")
include("common/simulate.jl")
include("bayes_net.jl")
include("test_bayes_net.jl")
include("LifeDetectionPOMDP.jl")
include("states.jl")
include("actions.jl")
include("rewards.jl")
include("observations.jl")
include("transition.jl")

export
    LifeDetectionPOMDP, determineMaxObs, plot_alpha_action_heatmap, simulate_policyVLD # Can add more through commas
end

