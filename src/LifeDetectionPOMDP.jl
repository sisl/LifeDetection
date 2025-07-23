using POMDPs # TODO: check if i can delete this
using POMDPTools

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
	# k::Vector{Float64} 			# Cost of observations
	discount::Float64 # Discount factor
	MAX_PENALTY::Int64 # Maximum penalty for doing infeasible Actions
	std_fraction::Float64 # Std deviation fraction variability
end


# Custom constructor to handle dynamic initialization
function LifeDetectionPOMDP(;
	bn::DiscreteBayesNet, # Bayesian Network,
	test_bn::BayesNet, # test Bayes Net (Continuous)
	λ::Float64,
	τ::Float64,
	ACTION_CPDS::Dict,
	max_obs::Int64,
	inst::Int64=7, # number of instruments / not using instrument
	sample_volume::Int64=500,
	life_states::Int64=3,
	ACC_RATE::Int64=270,
	sample_use::Vector{Int64}=[1, 1, 1, 1, 1, 1, 1], # cost of observations
	# k::Vector{Float64} = [HRMS*10e6, SMS*10e6, μCE_LI*10e6, ESA*10e6, microscope*10e6, nanopore*10e6], # cost of observations
	discount::Float64=0.9,
	MAX_PENALTY::Int64=10000,
	std_fraction::Float64=0.25,
)
	return LifeDetectionPOMDP(bn, test_bn, λ, τ, ACTION_CPDS, max_obs, inst, sample_volume, life_states, ACC_RATE, sample_use, discount,MAX_PENALTY,std_fraction)
end

# 1 -> dead
# 2 -> alive
# 3 -> terminal state
POMDPs.states(pomdp::LifeDetectionPOMDP) = 1:(pomdp.sample_volume*pomdp.life_states+pomdp.life_states) #(pomdp.sample_volume*((2^pomdp.life_states)))+(2^pomdp.life_states)

# run sensor (2+i), where i the ith instrument, and the last instrument is doing nothing
# declare dead (second to last) declare alive (last action)
POMDPs.actions(pomdp::LifeDetectionPOMDP) = [1:pomdp.inst..., pomdp.inst+1, pomdp.inst+2]

# Extra observation at beginning which will be null observation
POMDPs.observations(pomdp::LifeDetectionPOMDP) = 1:((pomdp.max_obs)*(pomdp.sample_volume+1)) # pomdp.sample_volume*pomdp.life_states+pomdp.life_states+1 

POMDPs.stateindex(pomdp::LifeDetectionPOMDP, s::Int)  = s
POMDPs.actionindex(pomdp::LifeDetectionPOMDP, a::Int) = a
POMDPs.obsindex(pomdp::LifeDetectionPOMDP, o::Int)    = o


# TODO: do we want to start with different states? With different accumulations? (YES)
# state_to_stateindex(0, 1) # TODO: change in future, so it starts at any state
POMDPs.initialstate(pomdp::LifeDetectionPOMDP) = uniform_state_distribution(pomdp) #initialstateSample(pomdp, rand(0:100)) 
# POMDPs.initialstate(pomdp::LifeDetectionPOMDP) = initialstateSample(pomdp, 0) # 50% chance of being alive or dead with no starting sample    

function uniform_state_distribution(pomdp::LifeDetectionPOMDP)
	states = [s for s in 1:(pomdp.sample_volume * pomdp.life_states)
	          if !POMDPs.isterminal(pomdp, s)]
	probs = fill(1.0 / length(states), length(states))
	return SparseCat(states, probs)
end
function initialstateSample(pomdp::LifeDetectionPOMDP, sample_volume::Int)
	# Sample the life state from BN prior
	P_life = pomdp.bn.cpds[1].distributions[1].p[2]
	s1 = state_to_stateindex(sample_volume, 1)  # dead
	s2 = state_to_stateindex(sample_volume, 2)  # alive
	return SparseCat([s1, s2], [1 - P_life, P_life])
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

function POMDPs.transition(pomdp::LifeDetectionPOMDP, s::Int, a::Int)
	sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)

	# Declaration action → go to terminal state (life_state = 3)
	if a > pomdp.inst
		return Deterministic(state_to_stateindex(sample_volume, 3))
	end

	# Accumulation action → randomize life state (biotic or abiotic)
	if a == pomdp.inst
		if pomdp.std_fraction != 0
			# Sample multiple possible accumulation amounts with probabilities
			μ = pomdp.ACC_RATE
			σ = pomdp.ACC_RATE * pomdp.std_fraction
			max_volume = pomdp.sample_volume

			# Generate possible accumulation values (you can truncate this range if needed for performance)
			acc_vals = 0:max_volume - sample_volume

			# Compute probability of each accumulation using Normal(μ, σ)
			acc_probs = [pdf(Normal(μ, σ), a) for a in acc_vals]
			acc_probs = acc_probs ./ sum(acc_probs)  # Normalize

			# Compute new sample volumes
			sample_vols = [min(sample_volume + a, max_volume) for a in acc_vals]

			# Get life prior from Bayes net
			P_life = pomdp.bn.cpds[1].distributions[1].p[2]

			# Build support and probabilities for SparseCat
			support = Int[]  # will hold state indices
			probs = Float64[]  # will hold corresponding probabilities

			for (i, sv) in enumerate(sample_vols)
				p_acc = acc_probs[i]
				push!(support, state_to_stateindex(sv, 1))  # dead
				push!(support, state_to_stateindex(sv, 2))  # alive
				push!(probs, p_acc * (1 - P_life))  # dead
				push!(probs, p_acc * P_life)        # alive
			end	
			return SparseCat(support, normalize(probs,1.0))
		else
			sample_volume = min(sample_volume + pomdp.ACC_RATE, pomdp.sample_volume)

			# update life state randomly (based on BN prior)
			P_life = pomdp.bn.cpds[1].distributions[1].p[2]
			s1 = state_to_stateindex(sample_volume, 1)  # dead
			s2 = state_to_stateindex(sample_volume, 2)  # alive
			return SparseCat([s1, s2], [1 - P_life, P_life])
		end
	end

	# Instrument action → reduce sample volume
	sample_volume = clamp(sample_volume - pomdp.sample_use[a], 0, pomdp.sample_volume)

	# Stay in same life state while using instrument
	return Deterministic(state_to_stateindex(sample_volume, life_state))
end

function POMDPs.observation(pomdp::LifeDetectionPOMDP, a::Int, sp::Int)

	sample_volume, life_state = stateindex_to_state(sp, pomdp.life_states)

	# return null observation if...
	# terminal state
	# declare alive/dead
	# not using instrument

	# TODO: sample volume is 0 ? (I dont think this worked - GK)
	if POMDPs.isterminal(pomdp, sp) || a == pomdp.inst + 1 || a == pomdp.inst + 2 #|| sample_volume == 0
		return Deterministic(1)#state_to_stateindex(sample_volume, life_state))
	end

	if a == pomdp.inst
		obs_range = obs_sample_volume(sample_volume, pomdp.max_obs)
		probs = fill(1.0 / length(obs_range), length(obs_range))  # Uniform over all obs in that volume range
		return Deterministic(obs_range[1]) #SparseCat(obs_range, probs)
		# probs = zeros(pomdp.max_obs+1)
		# probs[end] = 1.0
		# return SparseCat(obs_sample_volume(sample_volume, pomdp.max_obs+1),probs)	
	end

	if sample_volume < pomdp.sample_use[a] # infeasible sample volume
		obs_range = obs_sample_volume(sample_volume, pomdp.max_obs)
		probs = fill(1.0 / length(obs_range), length(obs_range))
		return Deterministic(obs_range[1])
	end

	return SparseCat(obs_sample_volume(sample_volume, pomdp.max_obs), distObservations(pomdp.ACTION_CPDS, life_state, a, pomdp.max_obs))
end


function observation_simulate(pomdp::LifeDetectionPOMDP, a::Int, sp::Int)

	sample_volume, life_state = stateindex_to_state(sp, pomdp.life_states)

	# return null observation if...
	# terminal state
	# declare alive/dead
	# not using instrument

	# TODO: sample volume is 0 ? (I dont think this worked - GK)
	if POMDPs.isterminal(pomdp, sp) || a == pomdp.inst + 1 || a == pomdp.inst + 2 #|| sample_volume == 0
		return 1#state_to_stateindex(sample_volume, life_state))
	end

	if a == pomdp.inst
		obs_range = obs_sample_volume(sample_volume, pomdp.max_obs)
		probs = fill(1.0 / length(obs_range), length(obs_range))  # Uniform over all obs in that volume range
		return obs_range[1] #SparseCat(obs_range, probs)
		# probs = zeros(pomdp.max_obs+1)
		# probs[end] = 1.0
		# return SparseCat(obs_sample_volume(sample_volume, pomdp.max_obs+1),probs)	
	end

	if sample_volume < pomdp.sample_use[a] # infeasible sample volume
		obs_range = obs_sample_volume(sample_volume, pomdp.max_obs)
		probs = fill(1.0 / length(obs_range), length(obs_range))
		return obs_range[1]
	end

	posterior = infer(bn, pomdp.ACTION_CPDS[a] ,evidence=(Assignment(:sL => life_state )))
	samples = zeros(length(posterior.dimensions))

	for (i, obs_var) in enumerate(posterior.dimensions)
		if obs_var == :o4 || obs_var == :o6 || obs_var == :o7
			obs = rand(pomdp.test_bn.cpds[pomdp.test_bn.name_to_index[obs_var]].accessor((life_state == 2)))
			n_bins = length(pomdp.bn.cpds[pomdp.bn.name_to_index[obs_var]].distributions[life_state].p)
			lo = 0.0
			hi = 1.0
			bin_index = clamp(fld((obs - lo) / (hi - lo) * n_bins + 1, 1), 1, n_bins)
			samples[i] = bin_index
		else
			bin_index = rand(pomdp.test_bn.cpds[pomdp.test_bn.name_to_index[obs_var]].distributions[life_state])
			samples[i] = bin_index
		end
	end
	# samples contains bin indices for each observation variable
	sample_index = CartesianIndex(Int.(samples)...)  # convert to CartesianIndex
	domain_sizes = [length(bn.cpds[bn.name_to_index[v]].distributions[1].p) for v in posterior.dimensions]
	dims = Tuple(domain_sizes)
	obs_range = obs_sample_volume(sample_volume, pomdp.max_obs)
	offset = obs_range[1] - 1
	return LinearIndices(dims)[sample_index] + offset
end

function discretize_obs(obs_dict::Dict{Symbol, Float64}, bn, max_obs::Int)
    discretized = Dict{Symbol, Int}()

    for (var, val) in obs_dict
        val_clamped = clamp(val, 0.0, 0.9999)
        bin = clamp(floor(Int, val_clamped * max_obs) + 1, 1, max_obs)
        discretized[var] = bin
    end

    return discretized
end

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


function POMDPs.reward(pomdp::LifeDetectionPOMDP, s::Int, a::Int)
	sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)

	if a == pomdp.inst + 1  # declare abiotic
		return life_state == 1 ? -pomdp.τ*pomdp.λ : -pomdp.λ
	elseif a == pomdp.inst + 2  # declare biotic
		return life_state == 2 ? 0 : -pomdp.λ
	end

	if sample_volume < pomdp.sample_use[a] # infeasible sample volume
		return -pomdp.MAX_PENALTY
	end

	# sensing cost scaled by volume used
	# TODO: expected_change = expected_belief_change(pomdp, s, a)
	return -(1 - pomdp.λ) * (sample_volume/pomdp.sample_volume) #+ expected_change
end

function state_to_stateindex(sample_volume::Int, life_states::Int)
	return (sample_volume-1)*3+life_states+3
end

function obs_sample_volume(sample_volume::Int, max_obs::Int)
	return (max_obs*(sample_volume)+1):(max_obs*(sample_volume)+max_obs)
end

function stateindex_to_state(index::Int, n_life_states::Int)
	sample_volume = div(index - 1, n_life_states)
	life_state = mod(index - 1, n_life_states) + 1
	return sample_volume, life_state
end
