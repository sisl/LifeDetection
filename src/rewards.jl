
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
