module FluxDNN

using Flux
using Flux: onecold, onehotbatch
using Statistics
using Base.Iterators
using Random
using MLDatasets
using Parameters: @unpack
import Zygote

import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail

import Images

include("data.jl")
include("utils.jl")

include("models/mlp.jl")
include("models/lenet.jl")
include("models/tree.jl")

include("parse_args.jl")
end # module