# L2 regularization for Flux model
l2reg(m) = sum(x->sum(abs2, x), Flux.params(m))

# squared euclid distance of two models
l2diff(m1, m2) = sum(sum(abs2, w1 .- w2) for (w1,w2) in zip(params(m1),params(m2))) #* 2 / (l2reg(m1) + l2reg(m2))
	
function accuracy(dataset, m)
    num = sum(sum(onecold(m(x)|>cpu) .== onecold(y|>cpu) for (x,y) in dataset))
	den = sum(size(x, ndims(x)) for (x,y) in dataset)
	# @show dataset
	# @show typeof(dataset) length(dataset)
	# sum(size(x, ndims(x)) for (x,y) in dataset)
    return num / den
end

function multihingeloss(ŷ::AbstractMatrix, y::AbstractMatrix; margin = 1f0)
	ŷ = relu.(margin .+ ŷ .- 2 .* ŷ .* y)
	mean(ŷ .* ŷ) 
end

function sigmse(ŷ::AbstractMatrix, y::AbstractMatrix)
	ŷ = σ.(ŷ)
	sum(abs2, (ŷ .- y)) * 1 // length(y)
end

# number of observations
nobs(x) = size(x)[end]

# select the observations in `idxs`
function selectobs(x, idxs)
	xsize = size(x)[1:end-1]
	m = nobs(x)
	return reshape(reshape(x, : , m)[:, idxs], xsize..., :)
end

# extend Base hasfield since it works only on types
Base.hasfield(x, f) = hasfield(typeof(x), f)

"""
	gradloss(f, args...)

Return both the gradient and the value of `f`.
"""
function gradloss(f, args...)
	y, back = Zygote.pullback(f, args...)
	return back(Zygote.sensitivity(y)), y
end