## Data iteration utils.
# This is derived from the file data.jl in Knet
mutable struct Data{T}; 
    x
    y
    batchsize::Int
    length::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle::Bool
    xdevice
    ydevice
end

getdata(x::AbstractArray, ids) = x[(Base.Colon() for _=1:ndims(x)-1)..., ids]

# """
#     minibatch(x, [y], batchsize; shuffle, partial, xdevice, ydevice)

# Return an iterator of minibatches [(xi,yi)...] given data tensors x, y and batchsize.  
# The last dimension of x and y give the number of instances and should be equal. `y` is
# optional, if omitted a sequence of `xi` will be generated rather than `(xi,yi)` tuples.  Use
# `repeat(d,n)` for multiple epochs, `Iterators.take(d,n)` for a partial epoch, and
# `Iterators.cycle(d)` to cycle through the data forever (this can be used with `converge`).
# If you need the iterator to continue from its last position when stopped early (e.g. by a
# break in a for loop), use `Iterators.Stateful(d)` (by default the iterator would restart
# from the beginning).

# Keyword arguments:
# - `shuffle=false`: Shuffle the instances every epoch.
# - `partial=false`: If true include the last partial minibatch < batchsize.
# - `xdevice=cpu`: apply `xdevice(xbatch)` to each returned xbatch
# - `ydevice=cpu`: apply `ydevice(ybatch)` to each returned ybatch
# """
function minibatches(x,y,batchsize; shuffle=false,partial=false, xdevice=cpu, ydevice=cpu)
    nx = size(x)[end]
    nx != size(y)[end] && throw(DimensionMismatch("between x and y"))
    
    if nx < batchsize
        @warn "Number of data points less than batchsize, decreasing the batchsize to $nx"
        batchsize = nx
    end
    imax = partial ? nx : nx - batchsize + 1
    ids = 1:min(nx, batchsize)
    xt = typeof(getdata(x, ids) |> xdevice)
    yt = typeof(getdata(y, ids) |> ydevice)
    Data{Tuple{xt,yt}}(x,y,batchsize,nx,partial,imax,[1:nx;],shuffle,xdevice,ydevice)
end

function minibatches(x, batchsize; shuffle=false, partial=false, xdevice=cpu)
    nx = size(x)[end]
    imax = partial ? nx : nx - batchsize + 1
    @show nx
    # xtype may be underspecified, here we infer the exact types from the first batch:
    ids = 1:min(nx, batchsize)
    xt = typeof(getdata(x, ids) |> xdevice)
    Data{xt}(x, nothing, batchsize, nx, partial, imax, [1:nx;], shuffle,xdevice,nothing)
end

@propagate_inbounds function iterate(d::Data, i=0)     # returns data in d.indices[i+1:i+batchsize]
    if i >= d.imax
        return nothing
    end
    if d.shuffle && i == 0
        shuffle!(d.indices)
    end
    nexti = min(i + d.batchsize, d.length)
    ids = d.indices[i+1:nexti]
    xbatch = getdata(d.x, ids) |> d.xdevice
    if d.y === nothing
        return (xbatch,nexti)
    else
        ybatch = getdata(d.y, ids) |> d.ydevice
        return ((xbatch,ybatch), nexti)
    end
end

eltype(::Type{Data{T}}) where T = T

function length(d::Data)
    n = d.length / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

function rand(d::Data)
    i = rand(0:(d.length-d.batchsize))
    return iterate(d, i)[1]
end

# Give length info in summary:
Base.summary(d::Data) = "$(length(d))-element $(typeof(d))"

# IterTools.ncycle(data,n) for multiple epochs
# Base.Iterators.cycle(data) to go forever
# Base.Iterators.take(data,n) for partial epochs
# IterTools.takenth(itr,n) to report every n iterations