"""
Second Bayesian network for evaluating SARSOP policy performance.
"""


"""
sL  = life                           (Boolean)
o1  = polyelectrolyte                (Boolean)
o2  = cell membrane                  (Boolean)
o3  = autofluorescence               (Boolean)
o4  = Molecular Assembly Index >= 15 (R ∈ [0, 1])
o5  = Biotic Amino Acid Diversity    (Z ∈ [0, 22])
o6  = L:R Chirality                  (R ∈ [0, 1])
o7  = Salinity                       (R ∈ [0, 1])
o8  = CHNOPS                         (R ∈ [0, 1])
o9  = pH                             (R ∈ [0, 14])
o10 = Redox Potential [V]            (R ∈ [-0.5, 0])
"""
cont_bn = BayesNet()

# life (Boolean)
push!(cont_bn, DiscreteCPD(:sL, [0.9, 0.1]))

# polyelectrolyte (Boolean)
push!(cont_bn, DiscreteCPD(:o1, [:sL], [2], [
	Categorical([0.95, 0.05]),
	Categorical([0.15, 0.85]),
]))

# cell membrane (Boolean)
push!(cont_bn, DiscreteCPD(:o2, [:sL], [2], [
	Categorical([0.95, 0.05]),
	Categorical([0.2, 0.8]),
]))

# autofluorescence (Boolean)
push!(cont_bn, DiscreteCPD(:o3, [:sL], [2], [
	Categorical([0.95, 0.05]),
	Categorical([0.1, 0.9]),
]))

# MAI (continuous, Beta)
push!(cont_bn, FunctionalCPD{Beta}(:o4, [:sL], sl -> sl[1] ? Beta(2, 13) : Beta(2, 4)))

# Biotic Amino Acid Diversity (continuous, discretized)
push!(cont_bn, DiscreteCPD(:o5, [:sL], [2], [
	DiscreteBeta(1, 10, lo=0.0, hi=22.0, bins=23),
	DiscreteBeta(3, 1, lo=0.0, hi=22.0, bins=23),
]))

# Chirality (continuous, Beta)
push!(cont_bn, FunctionalCPD{Beta}(:o6, [:sL], sl -> sl[1] ? Beta(3, 3) : Beta(0.25, 0.25)))

# Salinity (continuous, Beta conditioned on o2)
push!(cont_bn, FunctionalCPD{Beta}(:o7, [:o2], o2 -> o2[1] ? Beta(1, 1) : Beta(1.05, 1)))

# CHNOPS (5 bins) – child of o4 (2 bins), o5 (23 bins)
o4_cpds = [
	DiscreteBeta(1 + i*0.1, 1.5, bins=5) for i in 0:4  # one for each o4 bin
]
o5_cpds = [DiscreteBeta(1 + i*0.001, 1, bins=5) for i in 0:22]
o8_cpds = Categorical[]
for i in 0:4, j in 0:22
	push!(o8_cpds, JointCategorical([o4_cpds[i+1], o5_cpds[j+1]]))
end
push!(cont_bn, DiscreteCPD(:o8, [:o4, :o5], [5, 23], o8_cpds))

# pH (14 bins), parents o1 (2), o5 (23)
o1_cpds = [
	DiscreteGaussian(9.0, 7.0, lo=0.0, hi=14.0, bins=14),
	DiscreteGaussian(10.0, 9.0, lo=0.0, hi=14.0, bins=14),
]
o5_cpds = [
	DiscreteGaussian(μ, σ, lo=0.0, hi=14.0, bins=14)
	for (μ, σ) in zip(range(10.0, stop=11.0, length=23), range(3.0, stop=7.0, length=23))
]
o9_cpds = Categorical[]
for i in 0:1, j in 0:22
	push!(o9_cpds, JointCategorical([o1_cpds[i+1], o5_cpds[j+1]]))
end
push!(cont_bn, DiscreteCPD(:o9, [:o1, :o5], [2, 23], o9_cpds))

# Redox Potential (10 bins), parent: o5 (23)
o10_cpds = [
	DiscreteBeta(3.0, 10.0 - 7.0 * i / 22, lo=-0.5, hi=0.0, bins=10)
	for i in 0:22
]
push!(cont_bn, DiscreteCPD(:o10, [:o5], [23], o10_cpds))


