using DrWatson
# @quickactivate "FluxDNN"  # activate project

using ProgressMeter
using OnlineStats
# import Flux, FluxDNN
include("../src/FluxDNN.jl")
import Flux
using Flux: gpu, cpu, onehotbatch, onecold, params
# using Flux
using Zygote: @nograd
using Parameters: @with_kw, @unpack
using Printf

using CUDA
# using CuArrays
# import CUDAapi
# if CUDAapi.has_cuda()
# 	using CuArrays
# 	CuArrays.allowscalar(false)
# end

@nograd onecold		# workaround https://github.com/FluxML/Flux.jl/issues/1020

@with_kw mutable struct Args
	model = "mlp_100"  # [mlp_400_200, conv, tree_2] (NN architecture)
	optim = "adam" 	   # [sgd, momentum, nesterov, adam]
	activation = :relu # activation function [:relu, :σ, :swish]
	seed = -1
	floss = "nll"      # [nll, hinge, sigmse]
	dataset = "mnist"  # [mnist, fashion]
	infotime = 2 	   # report every `infotime` epochs
	verb = 2            # verbosity
	cuda = true        # use cuda if available, otherwise run on cpu
	datapath = "~/data/" # path for your datasets
	pclass = -1        # if `pclass > 0` use only `pclass` training examples for each class
	η = 1e-3           # learning rate
	batchsize = 128
	ρ = 0              # L2 reg coefficient
	epochs = 100
end

main(; kws...) = main(Args(;kws...))

function main(args::Args)	
	args.seed > 0 && Random.seed!(args.seed)
	device = args.cuda ? gpu : cpu

    @info("Loading Dataset")
	xtrain, ytrain, xtest, ytest = FluxDNN.get_dataset(args)
	classes = sort(union(ytrain))
	ytrain, ytest = onehotbatch(ytrain, classes), onehotbatch(ytest, classes)
	dtrain = FluxDNN.minibatches(xtrain, ytrain, args.batchsize, shuffle=true, xdevice=device, ydevice=device)
	dtest = FluxDNN.minibatches(xtest, ytest, 1000, xdevice=device, ydevice=device)
	
    @info("Building Model")
	Net = FluxDNN.get_model_type(args)
	model =  Net() |> device
	ps = Flux.params(model)

	floss = FluxDNN.get_loss_function(args)
	opt = FluxDNN.get_optimizer(args)

	# opt_state = Flux.setup(opt, model)


	function report(epoch)
		train_acc = FluxDNN.accuracy(dtrain, model)*100
		test_acc = FluxDNN.accuracy(dtest, model)*100
		@info @sprintf("[%d]: Train acc: %.2f%%  Test  acc: %.2f%%", epoch, train_acc, test_acc)
	end


	@info("Start Training")
	report(0)
	for epoch in 1:args.epochs
		p = Progress(length(dtrain))
		avg_loss = Mean(weight=ExponentialWeight(0.1)) # moving average
		avg_acc = Mean(weight=ExponentialWeight(0.1)) # moving average
		
		# Flux.train!(model, dtrain, opt) do m, x, y
		# 	y_hat = m(x)
		# 	Flux.mse(y_hat, y)
		# end

		for (x, y) in dtrain

			gs = Flux.gradient(ps) do
				ŷ = model(x)
				l = floss(ŷ, y) + args.ρ*FluxDNN.l2reg(model) 
				
				fit!(avg_acc, mean(onecold(ŷ |>cpu) .== onecold(y|>cpu)))
				fit!(avg_loss, l)
				return l 
			end
			Flux.Optimise.update!(opt, ps, gs)

			# grads = Flux.gradient(model) do m
			# 	ŷ = m(x)
			# 	floss(ŷ, y) + args.ρ*FluxDNN.l2reg(model)
			# end

			# Flux.update!(opt_state, model, grads[1])

			# loss, grads = Flux.withgradient(model) do m
			# 	# Evaluate model and loss inside gradient context:
			# 	ŷ = m(x)
			# 	# Flux.crossentropy(ŷ, y)
			# 	Flux.mse(ŷ, y)
			# 	# floss(ŷ, y) + args.ρ*FluxDNN.l2reg(model) 
			# end

			# gs = Flux.gradient(model) do m
			# 	# ŷ = model(x)
			# 	ŷ = m(x)
			# 	l = floss(ŷ, y) + args.ρ*FluxDNN.l2reg(model) 
				
			# 	# fit!(avg_acc, mean(onecold(ŷ |>cpu) .== onecold(y|>cpu)))
			# 	# fit!(avg_loss, l)
			# 	return l 
			# end
			# Flux.Optimise.update!(opt, model, grads[1])

			# Flux.Optimise.update!(opt, model, gs)

			args.verb > 1 && ProgressMeter.next!(p, 
								showvalues = [(:loss, round(value(avg_loss), digits=4)), 
											(:accuracy, round(value(avg_acc), digits=4))],
								valuecolor=:red)
		end

		args.verb > 0 && epoch % args.infotime == 0 && report(epoch)
	end
end
