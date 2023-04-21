# imgsize has to be a 3d shape
function LeNet(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size =  (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 50)
    Chain(
            x -> reshape(x, imgsize..., ndims(x) ∈ (1,3) ? 1 : size(x)[end]),
            Conv((5, 5), imgsize[end]=>20, relu),
            MaxPool((2,2)),
            Conv((5, 5), 20=>50, relu),
            MaxPool((2,2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 500, relu), 
            # x -> (println("xt=$(typeof(x)) xe=$(eltype(x))  xs=$(size(x))"); x),
            Dense(500, nclasses)
        )
end