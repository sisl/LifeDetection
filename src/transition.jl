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
