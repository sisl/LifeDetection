
function simulate_policyVLD(pomdp, policy; type="SARSOP", n_episodes=1, verbose=true, threshold_high= 0.999, threshold_low=0.001)

	if verbose
		println("--------------------------------START EPISODES---------------------------------")
	end


	total_episode_rewards = zeros(Float64,n_episodes)
	accuracy = zeros(Float64,n_episodes)


	for episode in range(1, n_episodes)

		updater = DiscreteUpdater(pomdp)
		b = initialize_belief(updater, initialstateSample(pomdp, 0))
		s = initialstateSample(pomdp, 0)

		if verbose
			println("\nPolicy Simulation: Episode ", episode)
			# println("Step | Action       | Observation | Belief(Life) | True State | Acc Sample | Total Reward ")

			println("Step | Action        | Belief(Life) | True State | Acc Sample | Total Reward ")
			println("-------------------------------------------------------------------------------")
		end

		# Trackers
		step = 1
		true_state = 1 # track s_L (life state)
		a = 0
		s = 1
		sp = 1
		belief_life = pdf(b, 2)
		action_name = ""
		action_final = 0
		# belief_life = pdf(b, 2) # setting belief for life with no sample volume at first

		# metrics
		total_reward = 0.0
		acc = 0
		correct = Dict(
			"tt" => 0,  # true positive: declared life when life is true
			"tf" => 0,  # false positive: declared life when life is false
			"ft" => 0,  # false negative: declared dead when life is true
			"ff" => 0   # true negative: declared dead when life is false
		)

		# only for conops:
		modeAcc = true
		prevAction = 0

		while step <= 200 # always 200 steps  ### !isterminal(pomdp, s) &&

			if isterminal(pomdp, s)

				if action_final == 2 && true_state == 2
					correct["tt"] += 1  # true positive
				elseif action_final == 1 && true_state == 2
					correct["tf"] += 1  # false positive
				elseif action_final == 2 && true_state == 1
					correct["ft"] += 1  # false negative
				elseif action_final == 1 && true_state == 1
					correct["ff"] += 1  # true negative
				end

				
				# Get the current sample volume from state index
				sample_volume, _ = stateindex_to_state(s, pomdp.life_states)

				# Sample new life state while keeping sample_volume fixed
				s = rand(initialstateSample(pomdp, sample_volume))

				# Reinitialize belief for same sample_volume
				b = initialize_belief(updater, initialstateSample(pomdp, sample_volume))
				belief_life = pdf(b, state_to_stateindex(sample_volume, 2))

                if verbose
                    println("[Resetting] Reached terminal state. Sampling new initial state.")
                end
			end

			# get action, next state, and observation
			if type == "SARSOP"
				a = action(policy, b)
			elseif type == "greedy"
				a = action_greedy_policy(policy, b, step)
			elseif type == "CONOPS"
				a, modeAcc, prevAction = conops_orbiter(pomdp, s, modeAcc, prevAction, belief_life, threshold_high, threshold_low)
			end

			sp = rand(transition(pomdp, s, a))
			o = observation_simulate(pomdp, a, sp)#rand(observation(pomdp, a, sp))

			# Get reward and accumulate total reward
			r = reward(pomdp, s, a, sp)
			total_reward += r

			# format action and observation names
			action_name = a >= pomdp.inst+1 ? (a == pomdp.inst+1 ? "Declare Dead" : "Declare Life") : (a == pomdp.inst ? "Accumulate" : "Sensor $(a)")

			accu, true_state = stateindex_to_state(s, pomdp.life_states)  # Save the current state before transitioning 
			accu_2, true_state2 = stateindex_to_state(sp, pomdp.life_states)  # Save the current state before transitioning 
			s_check = s
			if true_state == 1 #&& step > 1
				s_check = s_check + 1
			end
			belief_life = pdf(b, s_check)
			
			if verbose
				@printf("%3d  | %-12s | %.3f        | %d          |  %d         | %.2f         \n",
					step, action_name, belief_life, true_state, accu, total_reward)
				println(accu_2)
			end
			
			Dict(
				"step" => step,
				"action" => a,
				"beliefLife" => belief_life,
				"trueState" => true_state,
				"accu" => acc,
				"totalReward" => total_reward,
				"observation" => o,
				"state" => s,
				"nextState" => sp,
				"belief" => b,
			)

			b = update(updater, b, a, o)

			# end
			s = sp
			step += 1
			action_final = a-pomdp.inst

			if true_state == 2 && a ==7
				correct["tf"] += 1  # false positive
			end
			if true_state == 1 && a ==7
				correct["ft"] += 1  # false positive
			end
		end

		println(correct)

		# Calculate accuracy using the correct dictionary
		total = correct["tt"] + correct["tf"] + correct["ft"] + correct["ff"]
		if total > 0
			acc = (correct["tt"] + correct["ff"]) / total
		else
			acc = 0.0
		end
		
		total_episode_rewards[episode] = total_reward
		accuracy[episode] = acc

	end

	if verbose
		println("--------------------------------END EPISODES---------------------------------")
		println("Average Rewards:", mean(total_episode_rewards))
		println("Average Accuracy:", mean(accuracy))
		println(accuracy)
	end

	return mean(total_episode_rewards), mean(accuracy)
end


function action_greedy_policy(policy, b, step)
    # Only two steps in greedy policy
    # 1. we use the sensor once, (or whatever the initial policy chooses)
    # 2. if the simulation isn't terminated, we declare whether or not it's alive or dead.
    
    if step == 1
        return action(policy, b)
    else
        return pdf(b, 2) ≤ 0.5 ? 1 : 2
    end
end

function conops_orbiter(pomdp, s, mode_acc, prev_action, belief_life, threshold_high, threshold_low)
    # Constants
    declare_dead_action = pomdp.inst + 1  # 8
    declare_life_action = pomdp.inst + 2  # 9
    accumulate_action = pomdp.inst        # 7

    # Get current sample volume
    sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)

    # Declare if confident enough
    if belief_life >= threshold_high
        return (declare_life_action, true, 0)
    elseif belief_life <= threshold_low
        return (declare_dead_action, true, 0)
    end

	if mode_acc
        return sum(pomdp.sample_use[1:5]) > sample_volume ? (7, true, 0) : (1, false, 1)
    end

    if prev_action < 5 && pomdp.sample_use[prev_action+1] < sample_volume
        return (prev_action + 1, false, prev_action + 1)
    end

    return (7, true, 0)
end