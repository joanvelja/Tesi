## FluxDNN.jl
Deep learning scripts, examples and utility functions on top of [Flux.jl](https://github.com/FluxML/Flux.jl) and julia.

### Usage
First you will have to clone the repo: 
```
git clone https://gitlab.com/bocconi-artlab/FluxDNN.git
```
The run the main script, `dnn.jl`,  to train your neural network. 
Let's train a multi-layer perceptron with 2 hidden layer of 100 neurons each on the MNIST dataset:
```julia
julia> include("scripts/dnn.jl")
Activating environment at `~/FluxDNN/Project.toml`

julia> main(model="mlp_100_100", dataset="mnist", batchsize=128, η=1e-3, epochs=100)
[ Info: Loading data set
[ Info: Building model
[ Info: Start training
[ Info: [0]: Train acc: 10.49%  Test  acc: 10.19%
Progress: 100%|███████████████████████████████████████████████████████| Time: 0:00:34
  loss:  0.27503902
Progress: 100%|███████████████████████████████████████████████████████| Time: 0:00:33
  loss:  0.13683645
[ Info: [2]: Train acc: 97.09%  Test  acc: 96.28%
...
```

