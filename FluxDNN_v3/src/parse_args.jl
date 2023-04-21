"""
This will output an x-tensor for train and one for test
and 1d y-tensors for train and  test.
"""
function get_dataset(args)
	@unpack dataset = args
	if hasfield(args,:datapath) && (args.datapath!==nothing)
		datapath = expanduser(args.datapath)
	else
		datapath = nothing
	end
	
	if dataset == "mnist"
		xsize = (28, 28, 1)
		dir = datapath===nothing ? nothing : joinpath(datapath, "MNIST") 
		xtrain, ytrain = MNIST.traindata(dir=dir)
		xtest, ytest = MNIST.testdata(dir=dir)
	elseif dataset == "fashion"
		xsize = (28, 28, 1)
		dir = datapath===nothing ? nothing : joinpath(datapath, "FashionMNIST") 
		xtrain, ytrain = FashionMNIST.traindata(dir=dir)
		xtest, ytest = FashionMNIST.testdata(dir=dir)
	elseif dataset == "cifar10"
		xsize = (32, 32, 3)
		dir = datapath===nothing ? nothing : joinpath(datapath, "CIFAR10") 
		xtrain, ytrain = CIFAR10.traindata(dir=dir)
		xtest, ytest = CIFAR10.testdata(dir=dir)
	elseif dataset == "cifar100"
		xsize = (32, 32, 3)
		dir = datapath===nothing ? nothing : joinpath(datapath, "CIFAR100") 
		xtrain, ytrain = CIFAR100.traindata(dir=dir)
		xtest, ytest = CIFAR100.testdata(dir=dir)
    else
        error("no such dataset: $(dataset)")
	end
	
	xtrain = reshape(xtrain, xsize..., :)
	xtest = reshape(xtest, xsize..., :)
	
	if hasfield(args, :pclass) && args.pclass > 0
		# Use only pclass examples for each class
	    iclass = []
		for c in union(ytrain) # for each class
			idx = findall(==(c), ytrain)
			shuffle!(idx)
			push!(iclass, idx[1:args.pclass])
		end
		idxs = shuffle!(vcat(iclass...))
		xtrain = selectobs(xtrain, idxs)
		ytrain = ytrain[idxs]
	end
	
    return xtrain, ytrain, xtest, ytest
end

function get_model_type(args)
	@unpack dataset, model, activation  = args
	
	nclasses =  hasfield(typeof(args), :nclasses) && args.nclasses > 0 ? args.nclasses :
				dataset in ["mnist","fashion","cifar10"] ? 10 :
		        dataset in ["cifar100"] ? 100 : error("specify number of classes")
	
	xsize =  dataset in ["mnist","fashion"] ? (28, 28, 1) :
		     dataset in ["cifar10", "cifar100"] ? (32, 32, 3) : error("TODO xsize")

	act = @eval $activation

	if startswith(model, "mlp")
		nhs = [parse(Int, h) for h in split(model, '_')[2:end]]
        Net = () -> MLP(prod(xsize), nhs, nclasses, act)
    elseif startswith(model, "tree")
		nhs = [parse(Int, h) for h in split(model, '_')[2:end]]
        Net = () -> TreeMLP(prod(xsize), nhs, nclasses, act)
	elseif model == "lenet"
        Net = () -> LeNet(imgsize=xsize, nclasses=nclasses)
	end
	return Net
end

function get_loss_function(args)
	@unpack floss = args
	
	loss = floss == "nll" ? (ŷ, y) -> Flux.logitcrossentropy(ŷ, y) :
		   floss == "hinge" ? (ŷ, y) -> multihingeloss(ŷ, y) :
		   floss == "sigmse" ? (ŷ, y) -> sigmse(ŷ, y) :
		   error("no such loss")
	return loss
	
end

function get_optimizer(args)
	@unpack optim, η = args

	opt = optim == "sgd" ? Descent(η) :
		  optim == "momentum" ? Momentum(η) :
		  optim == "nesterov" ? Nesterov(η) :
		  optim == "adam" ? ADAM(η) :
		  error("no optimizer \"$optim\"")
	return opt
	
end