"""
Second Bayesian network for evaluating SARSOP policy performance.
"""

using BayesNets
include("common/bayes_net_helpers.jl")

"""
sL  = life                           (Boolean)
o1  = polyelectrolyte                (Boolean)
o2  = cell membrane                  (Boolean)
o3  = autofluorescence               (Boolean)
o4  = Molecular Assembly Index >= 15 (Boolean)
o5  = Biotic Amino Acid Diversity    (Z ∈ [0, 22])
o6  = L:R Chirality                  (R ∈ [0, 1])
o7  = Salinity                       (R ∈ [0, 1])
o8  = CHNOPS                         (R ∈ [0, 1])
o9  = pH                             (R ∈ [0, 14])
o10 = Redox Potential [V]            (R ∈ [-0.5, 0])
"""
bn = BayesNet()

# life: Boolean
push!(bn, DiscreteCPD(:sL, [0.9, 0.1])) # null hypothesis

# polyelectrolyte: Boolean
push!(
	bn,
	DiscreteCPD(:o1, [:sL], [2], [
		Categorical([0.95, 0.05]),  # dead
		Categorical([0.15, 0.85]),   # alive
	]),
)

# cell membrane: Boolean
push!(
	bn,
	DiscreteCPD(:o2, [:sL], [2], [
		Categorical([0.95, 0.05]),  # dead
		Categorical([0.2, 0.8]),     # alive
	]),
)

# autofluorescence: Boolean
push!(
	bn,
	DiscreteCPD(:o3, [:sL], [2], [
		Categorical([0.95, 0.05]),  # dead
		Categorical([0.1, 0.9]),     # alive
	]),
)

# molecular assembly index >= 15: Boolean
push!(bn, DiscreteCPD(:o4, [:sL], [2], [
	Categorical([0.7, 0.3]),  # dead
	Categorical([0.2, 0.8]),   # alive
]))

# biotic amino acid diversity: 23 bins
push!(
	bn,
	DiscreteCPD(
		:o5,
		[:sL],
		[2],
		[
			DiscreteBeta(1, 10, lo=0.0, hi=22.0, bins=23), # dead
			DiscreteBeta(3, 1, lo=0.0, hi=22.0, bins=23),   # alive
		],
	),
)

# chirality: 10 bins
push!(bn, StaticCPD(:o6, [:sL], [2], [
	Beta(3, 3),     # dead
	Beta(0.5, 0.5), # alive
]))

# salinity: 10 bins
push!(
	bn,
	DiscreteCPD(:o7, [:o2], [2], [
		Beta(1, 1), # no cell membrane
		Beta(2, 1), # cell membrane
	]),
)

# CHNOPS: simplified approach - just depend on o4
push!(bn, StaticCPD(:o8, [:o4], [2], [
	Beta(1, 2),  # P(o8 | o4 = 0) (MA < 15)
	Beta(3, 1),  # P(o8 | o4 = 1) (MA ≥ 15)
]))
#plot_categorical(o8_cpds[30], labels=range(0,1,10), title="CHNOPS Distribution (o4=0, o5=0)")

# pH: simplified approach - just depend on o1
push!(bn, StaticCPD(:o9, [:o1], [2], [
	Normal(9.0, 7.0),  # P(o9 | o1 = 0) (no polyelectrolyte)
	Normal(10.0, 9.0), # P(o9 | o1 = 1) (polyelectrolyte present)
]))

# redox potential: 10 bins
o5_cpds = [DiscreteBeta(3.0, 10.0 - 7.0 * i / 22, lo=-0.5, hi=0.0, bins=10) for i ∈ 0:22]
push!(bn, DiscreteCPD(:o10, [:o5], [23], o5_cpds))
