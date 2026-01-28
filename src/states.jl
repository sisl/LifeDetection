# 1 -> dead
# 2 -> alive
# 3 -> terminal state
POMDPs.states(pomdp::LifeDetectionPOMDP) = 1:(pomdp.sample_volume*pomdp.life_states+pomdp.life_states) #(pomdp.sample_volume*((2^pomdp.life_states)))+(2^pomdp.life_states)

POMDPs.stateindex(pomdp::LifeDetectionPOMDP, s::Int)  = s

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
