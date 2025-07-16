using POMDPs # TODO: check if i can delete this
using POMDPTools
using MOMDPs

struct LifeDetectionMOMDP <: MOMDP{Int, Int, Int, Int}  # MOMDP{State_x, State_y , Action, Observation}
	bn::DiscreteBayesNet # Bayesian Network,
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
function LifeDetectionMOMDP(;
	bn::DiscreteBayesNet, # Bayesian Network,
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
	return LifeDetectionMOMDP(bn, λ, τ, ACTION_CPDS, max_obs, inst, sample_volume, life_states, ACC_RATE, sample_use, discount,MAX_PENALTY,std_fraction)
end

# MOMDPs.states(momdp::LifeDetectionMOMDP) = 1:(momdp.sample_volume*momdp.life_states+momdp.life_states) #(momdp.sample_volume*((2^momdp.life_states)))+(2^momdp.life_states)
MOMDPs.states_x(momdp::LifeDetectionMOMDP) = 1:momdp.sample_volume+1 #+1 is just the zero

# 1 -> dead
# 2 -> alive
# 3 -> terminal state
MOMDPs.states_y(momdp::LifeDetectionMOMDP) = 1:momdp.life_states 

# run sensor (2+i), where i the ith instrument, and the last instrument is doing nothing
# declare dead (second to last) declare alive (last action)
POMDPs.actions(momdp::LifeDetectionMOMDP) = [1:momdp.inst..., momdp.inst+1, momdp.inst+2]

# Extra observation at beginning which will be null observation
POMDPs.observations(momdp::LifeDetectionMOMDP) = 0:momdp.max_obs

MOMDPs.stateindex_x(momdp::LifeDetectionMOMDP, s_x::Int)  = s_x # s::Tuple{Int, Int}
MOMDPs.stateindex_y(momdp::LifeDetectionMOMDP, s_y::Int)  = s_y # s::Tuple{Int, Int}
POMDPs.actionindex(momdp::LifeDetectionMOMDP, a::Int) = a
POMDPs.obsindex(momdp::LifeDetectionMOMDP, o::Int)    = o+1

MOMDPs.initialstate_x(momdp::LifeDetectionMOMDP) = SparseCat(1:momdp.sample_volume, fill(1.0 / momdp.sample_volume, momdp.sample_volume))
MOMDPs.initialstate_y(momdp::LifeDetectionMOMDP, x) = SparseCat(1:2, [0.5,0.5])
# function MOMDPs.initialstate_y(momdp::LifeDetectionMOMDP, x) 
# 	P_life = momdp.bn.cpds[1].distributions[1].p[2]
# 	return SparseCat(1:2, [1.0 - P_life, P_life])
# end

MOMDPs.is_y_prime_dependent_on_x_prime(momdp::LifeDetectionMOMDP) = false
MOMDPs.is_x_prime_dependent_on_y(momdp::LifeDetectionMOMDP) = false
MOMDPs.is_initial_distribution_independent(momdp::LifeDetectionMOMDP) = true

POMDPs.isterminal(momdp::LifeDetectionMOMDP, s::Tuple{Int,Int}) = s[2] == 3
POMDPs.discount(momdp::LifeDetectionMOMDP) = momdp.discount

function MOMDPs.transition_x(momdp::LifeDetectionMOMDP, s::Tuple{Int,Int}, a::Int)
    x = s[1]

	# Declaration action → go to terminal state (life_state = 3)
	if a > momdp.inst
		return Deterministic(x)
	end

	# Accumulation action → randomize life state (biotic or abiotic)
	if a == momdp.inst

		# Sample multiple possible accumulation amounts with probabilities
		μ = momdp.ACC_RATE
		σ = momdp.ACC_RATE * momdp.std_fraction
		max_volume = momdp.sample_volume

		# Generate possible accumulation values (you can truncate this range if needed for performance)
		acc_vals = 1:max_volume+1 - x

		if x == 101 || x == 100
            return Deterministic(101)
        end

		# Compute probability of each accumulation using Normal(μ, σ)
		acc_probs = [pdf(Normal(μ, σ), a) for a in acc_vals]
		total_prob = sum(acc_probs)

        # # Handle near-zero or zero total probability
        # if total_prob < 1e-8
        #     return Deterministic(x)
        # end

		acc_probs = acc_probs ./ sum(acc_probs)  # Normalize

		# Compute new sample volumes
		sample_vols = [min(x + a, max_volume+1) for a in acc_vals]

		return SparseCat(sample_vols, acc_probs)

	end

	if momdp.sample_use[a] < x
		# Instrument action → reduce sample volume
		return Deterministic(clamp(x - momdp.sample_use[a], 1, momdp.sample_volume))
	end

	# Stay in same life state while using instrument if invalid action
	return Deterministic(x)


end

function MOMDPs.transition_y(momdp::LifeDetectionMOMDP, s::Tuple{Int,Int}, a::Int, x_prime::Int)
    y = s[2]
	
	# Declaration action → go to terminal state (life_state = 3)
	if a > momdp.inst
		return Deterministic(3)
	end

	# Accumulation action → randomize life state (biotic or abiotic)
	if a == momdp.inst
		P_life = momdp.bn.cpds[1].distributions[1].p[2]
		return SparseCat(1:2, [1 - P_life, P_life])
	end

	# Stay in same life state while using instrument
	return Deterministic(y)

end

function POMDPs.observation(momdp::LifeDetectionMOMDP, a::Int, s::Tuple{Int,Int})

	# return null observation if...
	# terminal state
	# declare alive/dead

	if POMDPs.isterminal(momdp, s) || a == momdp.inst + 1 || a == momdp.inst + 2
		return Deterministic(0)
	end

	if a == momdp.inst
		return Deterministic(0) #SparseCat(1:momdp.max_obs, fill(1.0 / momdp.max_obs, momdp.max_obs))
	end

	return SparseCat(1:momdp.max_obs, distObservations(momdp.ACTION_CPDS, s[2], a, momdp.max_obs))
end


function POMDPs.reward(momdp::LifeDetectionMOMDP, s::Tuple{Int,Int}, a::Int)

	if a == momdp.inst + 1  # declare abiotic
		return s[2] == 1 ? -momdp.τ*momdp.λ : -momdp.λ
	elseif a == momdp.inst + 2  # declare biotic
		return s[2] == 2 ? 0 : -momdp.λ
	end

	if s[1] < momdp.sample_use[a] # infeasible sample volume
		return -momdp.MAX_PENALTY
	end

	# sensing cost scaled by volume used
	# TODO: expected_change = expected_belief_change(momdp, s, a)
	return -(1 - momdp.λ) * (s[1]/momdp.sample_volume) #+ expected_change
end
