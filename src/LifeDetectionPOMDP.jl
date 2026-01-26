
##################### General Parameters for Instrument Sample Usage #############################

NUM_INSTRUMENTS = 7 # One extra for accumulate, Wouldn't change unless you change Bayes Net and Action CPDS
LIFE_STATES = 3

HRMS = 1#0 # 0.5e-6 # mL # organic compounds, just going to set to zero its too small
SMS_1 = 5 #400 # μL # 0.4 # mL # amino acid characerization
SMS_2 = 1   #100 # μL # 0.1 mL # Lipid Characterization
SMS = SMS_1 + SMS_2

μCE_LIF = 1                   #15 # μL #0.015 # mL # amino acid and lipid characterization
ESA_1 = 1#15 # μL #0.015  # mL # macronutrients
ESA_2 = 1   #75 # μL #0.075  # mL # micronutrients
ESA_3 = 1#15 # μL #0.015  # mL # macronutrients
ESA = ESA_1 + ESA_2 + ESA_3

microscope = 1# μL # 0.001 # mL # polyelectrolyte
nanopore = 89#10000 # μL # 10  # mL cell like morphologies
none = 0#non-instrument actions

##################### Mapping Instrument Actions to sample characteristics #####################

ACTION_CPDS = Dict(
	1 => [:o5, :o7, :o8, :o10],   # HRMS ()
	2 => [:o5, :o6],              # SMS
	3 => [:o5, :o6],              # μoE_LIF
	4 => [:o7, :o8],              # ESA
	5 => [:o2, :o3],              # microscope
	6 => [:o1],                   # nanopore
)


struct LifeDetectionPOMDP <: POMDP{Int, Int, Int}  # POMDP{State, Action, Observation}
	bn::DiscreteBayesNet # Bayesian Network,
	test_bn::BayesNet # test Bayes Net (Continuous)
	λ::Float64# Parameter for penalty
	τ::Float64# Parameter for declaring abiotic
	ACTION_CPDS::Dict# Connecting actions to CPDS
	max_obs::Int64# Maximum observation count for observation generator
	inst::Int64 # Number of instruments + accumulation action
	sample_volume::Int64# Maximum Sample Volume in Storage Container
	life_states::Int64# Life States (3)
	ACC_RATE::Int64# Accumulation Rate
	sample_use::Vector{Int64}# Sample used by each of the instruments
	discount::Float64 # Discount factor
	MAX_PENALTY::Int64 # Maximum penalty for doing infeasible Actions
	std_fraction::Float64 # Std deviation fraction variability
end


# Custom constructor to handle dynamic initialization
function LifeDetectionPOMDP(;
	bn::DiscreteBayesNet=bn, # Bayesian Network,
	test_bn::BayesNet=cont_bn, # test Bayes Net (Continuous), used for rollouts
	λ::Float64= 0.935,
	τ::Float64=0.0,
	ACTION_CPDS::Dict=ACTION_CPDS,
	max_obs::Int64=determineMaxObs(ACTION_CPDS, bn),
	inst::Int64=7, # number of instruments / not using instrument
	sample_volume::Int64=500,
	life_states::Int64=3,
	ACC_RATE::Int64=270,
	sample_use::Vector{Int64}=[HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none], # cost of observations
	discount::Float64=0.9,
	MAX_PENALTY::Int64=10000,
	std_fraction::Float64=0.25,
)
	return LifeDetectionPOMDP(bn, test_bn, λ, τ, ACTION_CPDS, max_obs, inst, sample_volume, life_states, ACC_RATE, sample_use, discount,MAX_PENALTY,std_fraction)
end



function POMDPs.isterminal(pomdp::LifeDetectionPOMDP, s::Int)
	sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)
	if life_state == 3
		return true
	else
		return false
	end
end

POMDPs.discount(pomdp::LifeDetectionPOMDP) = pomdp.discount


# Can possibly Delete:
# TODO: incorporate expected change in belief
function expected_belief_change(pomdp::LifeDetectionPOMDP, a::Int)
	if a >= pomdp.inst  # not using instrument
		return 0.0
	end

	expected_change = 0.0

	# P(L=true)
	prior = infer(pomdp.bn, :sL).potential[2]

	# get nodes for this action
	nodes = pomdp.ACTION_CPDS[a]
	for node in nodes
		for obs in 1:pomdp.max_obs
			# P(L=true | obs)
			posterior = infer(pomdp.bn, :sL, evidence=Assignment(Dict(node => obs)))
			posterior_prob = posterior.potential[2]

			# P(obs)
			obs_prob = 0.0
			for life_state in 1:2
				obs_prob += pomdp.bn.cpds[pomdp.bn.name_to_index[node]].distributions[life_state].p[obs] *
							(life_state == 2 ? prior : (1 - prior))
			end

			# E[|P(L) - P(L|obs)|] = P(obs)*(P(L)-P(L|obs))
			expected_change += obs_prob * abs(prior - posterior_prob)
		end
	end
	return expected_change
end

