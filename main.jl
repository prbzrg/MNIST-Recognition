#!/bin/bash
#=
exec julia --startup-file=yes --handle-signals=yes --sysimage-native-code=yes --compiled-modules=yes --banner=yes --color=yes --history-file=yes --depwarn=yes --warn-overwrite=yes --optimize=3 -g 2 --project "${BASH_SOURCE[0]}" "$@"
=#

using
    DiffEqFlux,
    DifferentialEquations,
    Flux,
    Flux.Data,
    Flux.Losses,
    MLBase,
    Plots,
    ProgressMeter

n_iter = 2
make_plot = false

prep_x(x) = Float32.(cat(x..., dims=4))

train_data = DataLoader(
    (MNIST.images(:train),
        Flux.onehotbatch(MNIST.labels(:train), 0:9),),
    batchsize=32,
    shuffle=true,
    partial=true,
)
test_data = DataLoader(
    (MNIST.images(:test),
        Flux.onehotbatch(MNIST.labels(:test), 0:9),),
    batchsize=32,
    shuffle=true,
    partial=true,
)

imgsize = (28, 28, 1)
imgsize_afterconvs = (1, 1, 32)
nclasses = 10

mdl = Chain(
    prep_x, # (28, 28, 1, N)
    Conv((3, 3), imgsize[3] => 16, tanh), # (26, 26, 16, N)
    MaxPool((2, 2)), # (13, 13, 16, N)
    Conv((3, 3), 16 => 32, tanh), # (11, 11, 32, N)
    MaxPool((2, 2)), # (5, 5, 32, N)
    Conv((3, 3), 32 => 32, tanh), # (3, 3, 32, N)
    MaxPool((2, 2)), # (1, 1, 32, 32)
    flatten,
    BatchNorm(prod(imgsize_afterconvs), tanh),
    Dropout(1 / 8),
    Dense(prod(imgsize_afterconvs), 64, tanh),
    BatchNorm(64, tanh),
    Dropout(1 / 8),
    Dense(64, 32, tanh),
    BatchNorm(32, tanh),
    Dropout(1 / 8),
    NeuralODE(Chain(
        # BatchNorm(32, tanh),
        # Dropout(1/8),
        Dense(32, 16, tanh),
        # BatchNorm(16, tanh),
        # Dropout(1/8),
        Dense(16, 16, tanh),
        # BatchNorm(16, tanh),
        # Dropout(1/8),
        Dense(16, 32, tanh),
        # BatchNorm(32, tanh),
        # Dropout(1/8),
    ), 0.0:1.0, Tsit5()),
    x -> x.u[1],
    BatchNorm(32, tanh),
    Dropout(1 / 8),
    Dense(32, 16, tanh),
    BatchNorm(16, tanh),
    Dropout(1 / 8),
    Dense(16, nclasses, tanh),
    BatchNorm(nclasses, tanh),
    Dropout(1 / 8),
    softmax,
)

loss(x, y) = Flux.crossentropy(mdl(x), y)
loss(xy::Tuple) = loss(xy...)
loss(xy::DataLoader) = mean(loss.(xy))
accuracy(x, y) = recall(roc(Flux.onecold(y, 0:9), Flux.onecold(mdl(x), 0:9)))
accuracy(xy::Tuple) = accuracy(xy...)
accuracy(xy::DataLoader) = mean(accuracy.(xy))

opt = ADAM()
ps = Flux.params(mdl)

if make_plot
    loss_list_tn = Float64[]
    loss_list_tt = Float64[]
    acc_list_tn = Float64[]
    acc_list_tt = Float64[]

    function record_scores()
        push!(loss_list_tn, loss(first(train_data)))
        push!(loss_list_tt, loss(first(test_data)))
        push!(acc_list_tn, accuracy(first(train_data)))
        push!(acc_list_tt, accuracy(first(test_data)))

        display(plot(
            [acc_list_tn loss_list_tn acc_list_tt loss_list_tt],
            label=["train accuracy" "train loss" "test accuracy" "test loss"],
            title=["accuracy" "loss"],
            layout=2,
            legend=:outertopright,
            size=(1024, 512),
        ))
    end
end

@show accuracy(test_data)
make_plot && record_scores()
Flux.@epochs n_iter begin
    p = Progress(length(train_data))
    function cb()
        make_plot && record_scores()
        next!(p)
    end
    Flux.train!(loss, ps, train_data, opt, cb=cb)
    @show accuracy(test_data)
end
make_plot && savefig(plot(
    [acc_list_tn loss_list_tn acc_list_tt loss_list_tt],
    label=["train accuracy" "train loss" "test accuracy" "test loss"],
    title=["accuracy" "loss"],
    layout=2,
    legend=:outertopright,
    size=(1024, 512),
), "figs.png")
