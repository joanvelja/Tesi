module RobustDNN

using DrWatson
quickactivate(@__DIR__, "FluxDNN")
include(srcdir("utils.jl"))
include(srcdir("data.jl"))
include(srcdir("parse_args.jl"))

using Flux
using Flux.Tracker: data
using Printf
using Parameters: @with_kw, @unpack

import CUDAapi
CUDAapi.has_cuda() && using CuArrays

# replicate a minibatch iterator nrep times
function rep_minibatches(mbatches, nrep)
	ds = [deepcopy(mbatches) for _=1:nrep]
	return (([x for (x,y) in mbs], [y for (x,y) in mbs]) for mbs in zip(ds...))
end

@with_kw mutable struct Args
	# parameters
	model = "mlp_100"  # [mlp_400_200, conv] (NN architecture)
	algo = "sing"      # [sing, rep] (i.e. standard NN or replicated NN aka robust ensemble)
	optim = "sgd" 	   # [sgd, momentum, nesterov, adam]
	activation = :relu # activation function [:relu, :σ, :swish]
	seed = -1
	floss = "nll"      # [nll, hinge, sigmse]
	dataset = "mnist"  # [mnist, fashion]
	infotime = 1 	   # report every `infotime` epochs
	cuda = true        # use cuda if available, otherwise run on cpu
	datapath = nothing # path for your datasets. Defaults to .julia/datadeps/

	pclass = -1        # if `pclass > 0` use only `pclass` training examples for each class
	η = 0.01           # learning rate
	batchsize = 128
	ρ = 0              # L2 reg coefficient
	epochs = 100

	nrep = 5 			# number of replicas
	γ = 0.01 			# coupling among replicas
	dγ = 0.01 			# coupling multiplicative increase factor
	γmax = 0.01 		# coupling max value
	d0 = 0.             # fixed distance between replicas
end

main(; kws...) = main(Args(;kws...))

function main(args::Args)
	CUDAapi.has_cuda() && CuArrays.allowscalar(false)
	args.seed > 0 && Random.seed!(args.seed)
	device = args.cuda ? gpu : cpu

    @info("Loading data set")
	xtrain, ytrain, xtest, ytest = get_dataset(args)
	classes = sort(union(ytrain))
	ytrain = Flux.onehotbatch(ytrain, classes)
	ytest = Flux.onehotbatch(ytest, classes)
	dtrain = minibatches(xtrain, ytrain, args.batchsize, shuffle=true, xdevice=device, ydevice=device)
	dtest = minibatches(xtest, ytest, 1000, xdevice=device, ydevice=device)
	
    @info("Building model")
	Net = get_model_type(args)
	model =  Net() |> device

	floss = get_loss_function(args)
	loss(x,y) = floss(x, y, model)
	
	if args.algo == "rep"
		@unpack γ, dγ, γmax, d0 = args
		replicas = [Net() |> device for i in 1:args.nrep]
		repdtrain = rep_minibatches(dtrain, args.nrep)
		reploss(x, y) = d0 == 0 ? sum(floss(x[a], y[a], m)  + γ*0.5*l2diff(m, model) for (a,m) in enumerate(replicas)) :
								  sum(floss(x[a], y[a], m)  + γ*abs(0.5*l2diff(m, model)-d0) for (a,m) in enumerate(replicas))
	end
	
	opt = get_optimizer(args)

	function report(epoch)
		train_acc = accuracy(dtrain, model)*100
		test_acc = accuracy(dtest, model)*100
		@info @sprintf("[%d]: Train acc: %.2f%%  Test  acc: %.2f%%", epoch, train_acc, test_acc)
		if args.algo == "rep"
			@info "---- replica dists: $(data.([l2diff(m, model) for m in replicas])[1])"
            #@info "---- replica train accs: $([100*accuracy(dtrain, m) for m in replicas])"
            #@info "---- replica test accs: $([100*accuracy(dtest, m) for m in replicas])"
		end
	end

	@info("Start training")
	report(0)
	for epoch in 1:args.epochs
		if args.algo == "sing"
			Flux.train!(loss, params(model), dtrain, opt)
		elseif args.algo == "rep"
			Flux.train!(reploss, params(replicas..., model), repdtrain, opt)
            γ *=  1 + dγ
            γ = min(γmax, γ)
		else
			error("no such algorithm: $(args.algo)")
		end
		epoch % args.infotime == 0 && report(epoch)
	end
end

end #module
