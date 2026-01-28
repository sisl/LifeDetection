
# Extra observation at beginning which will be null observation
POMDPs.observations(pomdp::LifeDetectionPOMDP) = 1:((pomdp.max_obs)*(pomdp.sample_volume+1)) # pomdp.sample_volume*pomdp.life_states+pomdp.life_states+1 

POMDPs.obsindex(pomdp::LifeDetectionPOMDP, o::Int)    = o


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