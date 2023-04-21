# note that nin must be divisible by all h[i], i=1:end
TreeMLP(nin, h, nclasses, act=relu) = Chain(
        x -> reshape(x, :, size(x)[end]),
        Tree(nin, h[1], act),
        [Tree(h[i], h[i+1], act) for i=1:length(h)-1]...,
        Dense(h[end], nclasses)
        )
	
÷
"""
    Tree(in::Integer, out::Integer, σ = identity)

Creates a tree layer with parameters `W` and `b`.
W represented as a matrix of dimensions `out × block` where `out` is the size
of the output layer and `block = out ÷ in` is the size of the receptive field of a single neuron.
The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × batch_size` matrix. The out `y` will be a vector or batch of length `out`.
The input size has to be divisible by the output size. 

```julia
julia> t = FluxDNN.Tree(6, 3)
Tree(6, 3)

julia> t(ones(6))
3×1 Array{Float64,2}:
 -0.46642001159489155
  1.2705999463796616 
  1.8125775456428528 
```
"""
struct Tree{F,S,T}
  W::S
  b::T
  σ::F
end

Tree(W, b) = Tree(W, b, identity)

function Tree(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform, initb = zeros)
  @assert in % out == 0
  block = in ÷ out
  return Tree(initW(block, out), initb(out), σ)
end

Flux.@functor Tree

function (a::Tree)(x::AbstractArray)
	@assert size(x)[1] == prod(size(a.W))

	W, b, σ = a.W, a.b, a.σ
  block, out = size(W)
  x = reshape(x, block, out, :)
  out = reshape(sum(W .* x, dims=1), out, :)
  σ.(out)
end

function Base.show(io::IO, l::Tree)
  print(io, "Tree(", prod(size(l.W)), ", ", size(l.W, 2))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end