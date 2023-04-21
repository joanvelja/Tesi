MLP(nin, h, nclasses, act=relu) = Chain(
        x -> reshape(x, :, size(x)[end]),
        Dense(nin, h[1], act),
        [Dense(h[i], h[i+1], act) for i=1:length(h)-1]...,
        Dense(h[end], nclasses)
        )
		