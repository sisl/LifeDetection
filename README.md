# LifeDetection
Autonomy for the Enceladus Orbilander's proposed Life Detection instrument suite.

![orbilander](orbilander.png)

*[1] MacKenzie, S. M., et al., (2021), The Enceladus Orbilander Mission Concept: Balancing Return and Resources in the Search for Life, Planet. Sci. J, 2(77), doi: 10.3847/PSJ/abe4da.*


## Running Sweeps

After modifying the `sweep.yaml` files for different parameters, solve for policies with the set hyperparameters:
```
julia test/solve.jl
julia test/solve.jl <folder to filename >.yaml

# example:
julia test/solve.jl test/sweep.yaml
```