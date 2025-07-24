using Plots
using Graphs
using LaTeXStrings

function plot_alpha_action_heatmap(policy, pomdp_momdp,num_samples)
	if pomdp_momdp
		num_vectors = size(policy.alphas, 1)
		num_samples = num_samples +1
		belief_range = 1000
		b_vals = range(0, 1, length=belief_range)

		# Each row: sample volume, each column: belief in life
		dominating = zeros(Int, num_samples, belief_range)

		for (i, b) in enumerate(b_vals)
			for (j, s) in enumerate(1:num_samples)
				best_score = -Inf
				best_alpha = 0
				for k in 1:num_vectors
					α = policy.alphas[k]
					idx1 = (s - 1) * 3 + 1  # life = 1
					idx2 = (s - 1) * 3 + 2  # life = 2
					v = α[idx1]*(1-b) + α[idx2]*b
					if v > best_score
						best_score = v
						best_alpha = policy.action_map[k]
					end
				end
				dominating[j, i] = best_alpha
			end
		end
	else
		num_samples = length(policy.alphas)  # e.g., 101
		belief_range = 1000
		b_vals = range(0, 1, length=belief_range)

		# Each row: sample volume (alpha index), each column: belief in life=2
		dominating = zeros(Int, num_samples, belief_range)

		for (j, sam) in enumerate(1:num_samples)  # over sample indices
			alpha_set = policy.alphas[sam]  # list of alpha vectors at this sample index
			for (i, b) in enumerate(b_vals)  # over belief in life=2
				best_score = -Inf
				best_action = 0
				for (k, α) in enumerate(alpha_set)
					# Assuming α = [life=1, life=2, null]
					v = α[1]*(1 - b) + α[2]*b
					# Map action from policy.action_map if needed
					action = policy.action_map !== nothing ? policy.action_map[sam][k] : k
					if v > best_score
						best_score = v
						best_action = action
					end
				end
				dominating[sam, i] = best_action
			end
		end
	end

	# Define a discrete colormap with 9 colors
	# discrete_colors = cgrad(:Paired_12, 9, categorical=true)
	discrete_colors = cgrad([
		# 1–6: Paired_12 mostly, with 2 = teal and 6 = warm orange
		"#a6cee3",  # 1: light blue
		"#1ca3a3",  # 2: teal
		"#b2df8a",  # 3: light green
		"#33a02c",  # 4: mid green
		"#fb9a99",  # 5: light red-pink
		"#fdae61",  # 6: warm orange (changed from strong red)

		# 7: neutral-ish (slate)
		"#999999",  # 7: slate gray

		# 8 & 9: bold opposing red and blue
		"#d73027",  # 8: strong red
		"#4575b4",   # 9: strong blue
	], categorical=true)

	# Define tick positions in the *middle* of each discrete color block

	p = heatmap(
		1:num_samples, b_vals,dominating',
		xlabel="Current Accumulated "* L"s_V",
		ylabel = "Belief of Sample Biotic State: " * L"b(s_L)",
		# title="Belief in Life (P(life=1))",
		colorbar_title="Action to Take: " *L"\{a_1, ... a_9\} ",
		color = discrete_colors,
		clims = (0.5, 9.5),  # Important: avoids color blending
		fontfamily = "Computer Modern",
		guidefont = font(12, "Computer Modern"),    # Axis labels font size 20
		tickfont = font(12, "Computer Modern"),     # Tick labels font size 16
		legendfont = font(16, "Computer Modern"),   # Legend font size 18
		titlefont = font(20, "Computer Modern"),     # Title font size 24
		# background_color = :transparent,
		# foreground_color_subplot = :white,
		# guidefontcolor = :white,
		# tickfontcolor = :white,
		# size=(800, 400),
    	dpi=1000,
	)


	if !isdir("./figures")
		mkpath("./figures")
	end
	display(p)
	savefig(p, "./figures/plot_alpha_action_heatmap.pdf")
	return p

end
function plot_alpha_vectors_VLD(policy::AlphaVectorPolicy, pomdp, sample::Int)
    alpha_vectors = policy.alphas
    action_map = policy.action_map

    num_vectors = size(alpha_vectors, 1)
    b = range(0.0, 1.0, length=300)  # Belief in life (P(L=1))

    # Get indices in state vector for life and dead at this sample
    dead_idx = state_to_stateindex(sample, 1)
    life_idx = state_to_stateindex(sample, 2)

    # Compute value of each alpha vector at each belief
    V_matrix = zeros(num_vectors, length(b))
    for i in 1:num_vectors
        α = alpha_vectors[i]
        α_dead = α[dead_idx]
        α_life = α[life_idx]
        V_matrix[i, :] .= @. α_life * b + α_dead * (1 - b)
    end

    # Determine which vectors are optimal at *any* belief (up to numerical error)
    keep = falses(num_vectors)
    for j in 1:length(b)
        vj = V_matrix[:, j]
        max_val = maximum(vj)
        for i in 1:num_vectors
            if isapprox(vj[i], max_val; atol=1e-6)
                keep[i] = true
            end
        end
    end

    println("Kept $(count(keep)) of $num_vectors alpha vectors")

    # Plot only dominant vectors
    p = Plots.plot(title="Alpha Vectors at Sample Volume = $sample",
                   xlabel="Belief in Life (P(L=1))", ylabel="Value",
                   legend=:bottomleft)

    for i in 1:num_vectors
        if keep[i]
            α = alpha_vectors[i]
            α_dead = α[dead_idx]
            α_life = α[life_idx]
            V_b = @. α_life * b + α_dead * (1 - b)
            plot!(b, V_b, label="α_$i (a=$(action_map[i]))")
			println(V_b)
		end
    end

    display(p)
    savefig(p, "alpha_vectors_sample_$sample.png")
    return p
end




# TODO: Need to fix alpha plots, and Decision tree plots
function dominating_alphas(policy)#::AlphaVectorPolicy)
	num_vectors = size(policy.alphas, 1)
	num_samples = 101  # or infer from alpha size
	b_vals = range(0, 1, length=100)

	dom_idx_per_sample = Vector{Int}(undef, num_samples)

	for s in 1:num_samples
		best_alpha = 0
		best_score = -Inf

		for i in 1:num_vectors
			α = policy.alphas[i]
			# get index of life=1 and life=2 at this sample
			idx1 = (s - 1) * 3 + 1  # life = 1
			idx2 = (s - 1) * 3 + 2  # life = 2

			Vb = [α[idx1]*b + α[idx2]*(1-b) for b in b_vals]
			max_V = maximum(Vb)

			if max_V > best_score
				best_score = max_V
				best_alpha = i
			end
		end

		dom_idx_per_sample[s] = best_alpha
	end

	return dom_idx_per_sample
end
function plot_alpha_dots(policy)
	dominating = dominating_alphas(policy)


	alpha_actions = [α for α in policy.action_map]  # or however actions are stored
	action_colors = Dict(a => c for (a, c) in zip(unique(alpha_actions), palette(:tab10)))

	action_per_sample = [alpha_actions[i] for i in dominating]
	sample_range = 1:length(dominating)

	p = scatter(sample_range, dominating,
		group=action_per_sample,
		color=action_per_sample .|> x -> action_colors[x],
		xlabel="Sample Volume",
		ylabel="Dominating Alpha Index",
		title="Dominating Alpha Vector per Sample Volume",
		legend=:topright,
		markersize=4)


	if !isdir("./figures")
		mkpath("./figures")
	end
	display(p)
	savefig(p, "./figures/plot_alpha_dots.png")
	return p

end


function plot_decision_tree(tree_data::Tuple{Vector{String}, Dict{Tuple{Int64, Int64}, String}, Vector{Tuple{Int64, Int64}}})
	node_labels, edge_colors, edges = tree_data

	g = DiGraph(length(node_labels))
	for (i, j) in edges
		add_edge!(g, i, j)
	end

	t = TikzGraphs.plot(g, TikzGraphs.Layouts.Layered(), node_labels, edge_styles=edge_colors,
		node_style="draw", graph_options="nodes={draw,circle}")
	# NOTE: delete line '\setmainfont{Latin Modern Math}' to compile
	TikzPictures.save(TikzPictures.TEX("./figures/decision_tree.tex"), t)
	return t
end



function plot_belief_over_time(b_hist, a_hist; pomdp=nothing)
	action_labels = if pomdp === nothing
		string.(a_hist)
	else
		[a ≥ pomdp.inst+1 ? (a == pomdp.inst+1 ? "Declare Dead" : "Declare Life") :
		 (a == pomdp.inst ? "Accumulate" : "Sensor $(a)") for a in a_hist]
	end

	p = plot()
	for a_label in unique(action_labels)
		idxs = findall(==(a_label), action_labels)
		plot!(idxs, b_hist[idxs], label=a_label)
	end

	xlabel!("Step")
	ylabel!("P(life = 1)")
	title!("Belief in Life Over Time by Action")

	savefig(p, "./figures/belief_over_time.png")
	display(p)
	return p
end

function make_decision_tree(pomdp, policy; max_depth=3)
	node_labels = String[]
	edge_colors = String[]
	edges = Tuple{Int, Int}[]

	updater = DiscreteUpdater(pomdp)
	b0 = initialize_belief(updater, initialstate(pomdp))

	function traverse(b, parent_index::Union{Int, Nothing}=nothing, edge_color::Union{String, Nothing}=nothing, depth=1)
		a = action(policy, b)
		b_val = pdf(b, 2)
		action_name = a ≤ 2 ? (a == 1 ? "Dead" : "Alive") : "I$(a - 2)"
		label = "$(action_name), P(life)=$(round(b_val, digits=2))"
		push!(node_labels, label)
		current_index = length(node_labels)

		# record edge if a parent exists
		if parent_index !== nothing && edge_color !== nothing
			push!(edge_colors, edge_color)
			push!(edges, (parent_index, current_index))
		end

		# stop if we've reached max depth or if action is terminal
		if depth >= max_depth || a ≤ 2
			return
		end

		# branch for each observation
		for o in POMDPs.observations(pomdp)
			obs_name = o == 1 ? "red" : "green"
			b_new = update(updater, b, a, o)
			traverse(b_new, current_index, obs_name, depth+1)
		end
	end

	traverse(b0)
	edge_colors = Dict(zip(edges, edge_colors))
	return (node_labels, edge_colors, edges)
end
