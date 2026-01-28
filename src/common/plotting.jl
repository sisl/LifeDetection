
function plot_alpha_action_heatmap(policy,num_samples)
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