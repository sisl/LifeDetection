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
bn = DiscreteBayesNet()

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
p = [ # U-shaped replacement for Beta(0.5, 0.5)
	4.5,
	2.6,
	1.7,
	1.2,
	0.9,
	0.9,
	1.2,
	1.7,
	2.6,
	4.5,
]
p ./= sum(p)
c6 = Categorical(p)
push!(bn, DiscreteCPD(:o6, [:sL], [2], [
	DiscreteBeta(3, 3, bins=10),  # dead
	c6,                            # alive
]))

# salinity: 10 bins
push!(
	bn,
	DiscreteCPD(:o7, [:o2], [2], [
		DiscreteBeta(1, 1, bins=5),   # no cell membrane
		DiscreteBeta(1.2, 1, bins=5),    # cell membrane
	]),
)

# CHNOPS: 10 bins
# o4 x o5 = 2 x 23 = 46 distributions to fully describe CPD
o4_cpds = [
	DiscreteBeta(1, 1.5, bins=5),  # P(o8 | o4 = 0) (MA < 15)
	DiscreteBeta(2, 1, bins=5),   # P(o8 | o4 = 1) (MA ≥ 15)
]
# P(o8 | o5 = 0 to 22)
o5_cpds = [DiscreteBeta(1 + i*0.001, 1, bins=5) for i ∈ 0:22]
# Joint distributions: P(o8 | o4, o5) = P(o8 | o4) * P(o8 | o5)
o8_cpds = Categorical[]
for i ∈ 0:1
	for j ∈ 0:22
		joint = JointCategorical([o4_cpds[i+1], o5_cpds[j+1]])
		push!(o8_cpds, joint)
	end
end
push!(bn, DiscreteCPD(:o8, [:o4, :o5], [2, 23], o8_cpds))
#plot_categorical(o8_cpds[30], labels=range(0,1,10), title="CHNOPS Distribution (o4=0, o5=0)")

# pH: 15 bins
o1_cpds = [
	DiscreteGaussian(9.0, 7.0, lo=0.0, hi=14.0, bins=14),
	DiscreteGaussian(10.0, 9.0, lo=0.0, hi=14.0, bins=14),
]
o5_cpds = [
	DiscreteGaussian(μ, σ, lo=0.0, hi=14.0, bins=14) for
	(μ, σ) in zip(range(10.0, stop=11.0, length=23), range(3.0, stop=7.0, length=23))
]
o9_cpds = Categorical[]
for i ∈ 0:1
	for j ∈ 0:22
		joint = JointCategorical([o1_cpds[i+1], o5_cpds[j+1]])
		push!(o9_cpds, joint)
	end
end
push!(bn, DiscreteCPD(:o9, [:o1, :o5], [2, 23], o9_cpds))

# redox potential: 10 bins
o5_cpds = [DiscreteBeta(3.0, 10.0 - 7.0 * i / 22, lo=-0.5, hi=0.0, bins=10) for i ∈ 0:22]
push!(bn, DiscreteCPD(:o10, [:o5], [23], o5_cpds))
