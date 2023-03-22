

struct MultiLayerForwardData
    localfields::Array{Matrix{Float64}}
    activations::Array{Matrix{Float64}}
    inputs::Matrix{Float64}
end

"""
    forward(weights, ϕ, inputs)

Computes the forward pass of the network and returns the relevant results.
"""
function forward(
    weights::Array{Matrix{Float64}},
    ϕ::Function,
    inputs::Matrix{Float64}
)::MultiLayerForwardData

    localfields = []
    activations = []

    for (i, layer_weights) ∈ enumerate(weights)
        if i == 1
            layer_input = inputs
        else
            layer_input = activations[i-1]
        end

        push!(localfields, layer_input * layer_weights)
        push!(activations, ϕ.(localfields[i]))
    end

    return MultiLayerForwardData(localfields, activations, inputs)
end

"""
    ∇(dϕ, weights, inputs, data)

Compute the gradient of the network with respect to the weights.
"""
function ∇(
    dϕ::Function,
    weights::Array{Matrix{Float64}},
    data::MultiLayerForwardData,
    outputs::Array{Float64}
)::Array{Matrix{Float64}}
    # Compute the errors of the last layer
    errors = outputs - data.activations[end]

    # The local gradient for a neuron in layer j is defined as:
    # δj = dϕ(u[j]) * sum(δk * w[j,k] for k in the next layer)
    # where u[j] is the local field of the neuron in layer j and w[j,k] is the
    # weight of the connection between the neuron j and k in the next layer.
    # but for the last layer, δj = dϕ(u[j]) * (t[j] - y[j])
    # where t[j] is the target output and y[j] is the actual output.

    # So, we allocate the local gradients for the whole network and set them to 0
    local_gradients = [zeros(size(layer_weights)) for layer_weights ∈ weights]

    # We start with the last layer
    local_gradients[end] = dϕ.(data.localfields[end]) .* errors

    # Then we compute the local gradients for the rest of the layers using the
    # local gradients from the next layer
    for i ∈ length(weights)-1:-1:1
        local_gradients[i] = (dϕ.(data.localfields[i])
                              .*
                              (local_gradients[i+1] * weights[i+1]'))
    end

    # Having the local gradients, we can compute the gradient of the network
    # with respect to the weights, by simply multiplying the local gradient
    # with the input of the weight.

    # First we allocate the gradient, which is a matrix of the same size as the
    # weights
    gradient = [zeros(size(layer_weights)) for layer_weights ∈ weights]

    # Then we compute the gradient
    for layer ∈ 1:length(weights)
        if layer == 1
            gradient[layer] .= data.inputs' * local_gradients[layer]
        else
            gradient[layer] .= data.activations[layer-1]' * local_gradients[layer]
        end
    end

    return gradient
end

using Wandb
using LinearAlgebra
using Statistics

alinear(localfield) = localfield + 0.5
dlinear(localfield) = 1

asigmoid(localfield) = 1 / (1 + exp(-localfield))
dsigmoid(localfield) = asigmoid(localfield) * (1 - asigmoid(localfield))

atanh(localfield) = tanh(localfield)
dtanh(localfield) = 1 - tanh(localfield)^2

arelu(localfield) = max(0, localfield)
drelu(localfield) = localfield > 0 ? 1 : 0

"""
    errorandgradient(ϕ, dϕ, weights, data_inputs, data_outputs)

Computes the error and the gradient of the network with respect to the weights
for a given batch of data.
"""
function errorandgradient(ϕ, dϕ, weights, data_inputs, data_outputs)

    # Initialize the accumulators
    accum_gradient = [zeros(size(layer_weights)) for layer_weights ∈ weights]
    accum_error = 0

    # For each pattern
    for (input, target) ∈ zip(data_inputs, data_outputs)

        # Forward and backward pass
        forward_result = forward(weights, ϕ, input)
        pattern_gradient = ∇(dϕ, weights, forward_result, target)

        # Accumulate the error 
        errorsq = sum((target - forward_result.activations[end]) .^ 2)
        accum_error += errorsq / length(data_inputs)

        # Accumulate the gradient
        for i ∈ eachindex(accum_gradient, pattern_gradient)
            accum_gradient[i] += pattern_gradient[i] / length(data_inputs)
        end
    end

    return accum_error, accum_gradient
end

"""
    updateweights!(weights, gradient, learningrate)

Updates the weights of the network using the gradient and the learning rate.
"""
function updateweights!(weights, gradient, learningrate)
    for i ∈ eachindex(weights)
        weights[i] += learningrate * gradient[i]
    end
end

"""
    gradientnormsperlayer(gradient)

Computes the average gradient norm per layer.
"""
function gradientnormsperlayer(gradient)
    result = [0.0 for _ ∈ gradient]
    for (layer, layergrad) ∈ enumerate(gradient)
        layeravggrad = 0
        # display(layergrad)
        for neuron ∈ 1:size(layergrad)[2]
            layeravggrad += norm(layergrad[:, neuron])
        end
        result[layer] = layeravggrad / size(layergrad)[2]
    end
    return result
end

function predict(weights, ϕ, inputs)
    result = []
    for input ∈ inputs
        forward_result = forward(weights, ϕ, input)
        push!(result, forward_result.activations[end])
    end
    return result
end

function getlatentspace(weights, ϕ, inputs, numlayers)
    result = []
    for input ∈ inputs
        forward_result = forward(weights, ϕ, input)
        # the first layer is the input layer
        # The next numlayers are the processing layers
        # the next 1 layer is the compression layer
        compression_layer = numlayers + 2
        push!(result, forward_result.activations[compression_layer])
    end

    # Convert the result to a matrix
    result = vcat(result...)

    return result
end

using PyCall

function train(numlayers,
    numneurons,
    numinputs, numoutputs,
    hiddensize,
    learningrate, batchsize,
    epochs, activation,
    all_inputs, all_outputs,
    validation_inputs, validation_outputs,
    test_inputs, test_outputs; savepath=nothing)

    plt = pyimport("matplotlib.pyplot")
    wdbp = pyimport("wandb")

    # Compute descriptive name of the run
    run_name = "$(numlayers)l_$(numneurons)n_$(learningrate)" *
               "lr_$(batchsize)bs_$(epochs)e_$(activation)a"

    # Create wandb run
    lg = WandbLogger(
        project="trabajo2ia_autoencoders",
        entity="jpossaz",
        name=run_name,
        config=Dict(
            "numlayers" => numlayers,
            "numneurons" => numneurons,
            "numinputs" => numinputs,
            "numoutputs" => numoutputs,
            "learningrate" => learningrate,
            "hiddensize" => hiddensize,
            "batchsize" => batchsize,
            "epochs" => epochs,
            "activation" => activation,
            "data_size" => size(all_inputs)[1],
            "validation_size" => size(validation_inputs)[1],
            "test_size" => size(test_inputs)[1],
        )
    )

    # unpack inputs and outputs (dataframes) into arrays of row vectors
    unpackdata(data) = [Matrix(Vector(row)') for row in eachrow(data)]

    all_inputs = unpackdata(all_inputs)
    all_outputs = unpackdata(all_outputs)
    validation_inputs = unpackdata(validation_inputs)
    validation_outputs = unpackdata(validation_outputs)
    test_inputs = unpackdata(test_inputs)
    test_outputs = unpackdata(test_outputs)

    randweights(a, b) = rand(a, b) .* 2 .- 1

    weights = Vector{Matrix{Float64}}(vcat(
        [randweights(numinputs, numneurons)],
        [
            randweights(numneurons, numneurons) for i in 1:numlayers
        ],
        [randweights(numneurons, hiddensize)],
        [randweights(hiddensize, numneurons)],
        [
            randweights(numneurons, numneurons) for i in 1:numlayers
        ],
        [randweights(numneurons, numoutputs)]
    ))

    # display(numinputs)
    # display(numneurons)
    # display(numoutputs)
    # display(weights)

    if activation == :linear
        ϕ = alinear
        dϕ = dlinear
    elseif activation == :sigmoid
        ϕ = asigmoid
        dϕ = dsigmoid
    elseif activation == :tanh
        ϕ = atanh
        dϕ = dtanh
    elseif activation == :relu
        ϕ = arelu
        dϕ = drelu
    else
        error("Unknown activation function")
    end

    for epoch in 1:epochs

        dictforlogging = Dict{String,Any}(
            "epoch" => epoch,
        )

        (train_error, train_gradient) = errorandgradient(
            ϕ, dϕ, weights, all_inputs, all_outputs
        )
        # display(weights)
        updateweights!(weights, train_gradient, learningrate)

        # display(weights)
        dictforlogging["error/training"] = train_error
        # display(train_gradient)
        training_grad_norms = gradientnormsperlayer(train_gradient)
        for (layer, norm) in enumerate(training_grad_norms)
            dictforlogging["layer$(length(train_gradient) - layer)_grad/training"] = (
                norm
            )
        end
        dictforlogging["gradavg/training"] = mean(training_grad_norms)

        (validation_error, validation_gradient) = errorandgradient(
            ϕ, dϕ, weights, validation_inputs, validation_outputs
        )
        dictforlogging["error/validation"] = validation_error
        validation_grad_norms = gradientnormsperlayer(validation_gradient)
        for (layer, norm) in enumerate(validation_grad_norms)
            dictforlogging["layer$(length(validation_gradient) - layer)_grad/validation"] = (
                norm
            )
        end
        dictforlogging["gradavg/validation"] = mean(validation_grad_norms)

        (test_error, test_gradient) = errorandgradient(
            ϕ, dϕ, weights, test_inputs, test_outputs
        )
        dictforlogging["error/test"] = test_error
        test_grad_norms = gradientnormsperlayer(test_gradient)
        for (layer, norm) in enumerate(test_grad_norms)
            dictforlogging["layer$(length(test_gradient) - layer)_grad/test"] = (
                norm
            )
        end
        dictforlogging["gradavg/test"] = mean(test_grad_norms)

        # Compute error divergences
        dictforlogging["error/tra-val-divergence"] = (
            dictforlogging["error/validation"] / dictforlogging["error/training"])
        dictforlogging["error/tra-test-divergence"] = (
            dictforlogging["error/test"] / dictforlogging["error/training"])

        # Compute gradient divergences
        dictforlogging["gradavg/tra-val-divergence"] = (
            dictforlogging["gradavg/validation"] / dictforlogging["gradavg/training"])
        dictforlogging["gradavg/tra-test-divergence"] = (
            dictforlogging["gradavg/test"] / dictforlogging["gradavg/training"])

        if hiddensize == 2

            # Plot latent space
            space = getlatentspace(weights, ϕ, all_inputs, numlayers)

            fig, ax = plt.subplots()

            # Plot the latent space
            ax.scatter(
                space[:, 1],
                space[:, 2],
            )


            dictforlogging["latent_space"] = wdbp.Image(fig)

            # Close the figure so we don't run out of memory
            plt.close(fig)
        end
        Wandb.log(lg, dictforlogging)
    end
    close(lg)

    # Compute the latent spaces for the training, validation, and test sets
    train_space = getlatentspace(weights, ϕ, all_inputs, numlayers)
    validation_space = getlatentspace(weights, ϕ, validation_inputs, numlayers)
    test_space = getlatentspace(weights, ϕ, test_inputs, numlayers)

    # Build dataframes for the latent spaces
    train_space_df = DataFrame(
        [
            "x$(i)" => train_space[:, i] for i in axes(train_space, 2)
        ]...
    )

    validation_space_df = DataFrame(
        [
            "x$(i)" => validation_space[:, i] for i in axes(validation_space, 2)
        ]...
    )

    test_space_df = DataFrame(
        [
            "x$(i)" => test_space[:, i] for i in axes(test_space, 2)
        ]...
    )

    return (
        train_space_df,
        validation_space_df,
        test_space_df,
    )
end


function test_inference()
    alldata = loadbscdata("pancake_data.csv")

    Random.seed!(1234)

    # shuffle rows 
    alldata = alldata[randperm(size(alldata, 1)), :]

    # Split into training, validation and test sets
    trainingperc = 0.6
    validationperc = 0.2

    trainingend = floor(Int, trainingperc * size(alldata, 1))
    validationend = floor(Int, (trainingperc + validationperc) * size(alldata, 1))

    traindata = alldata[1:trainingend, :]
    validationdata = alldata[trainingend+1:validationend, :]
    testdata = alldata[validationend+1:end, :]
    println("Training set size: $(size(traindata, 1))")
    println("Validation set size: $(size(validationdata, 1))")
    println("Test set size: $(size(testdata, 1))")

    # first 2 columns are inputs, last two are outputs
    getinputsandoutputs(data) = (data[:, 6:11], data[:, 6:11])

    (traininputs, trainoutputs) = getinputsandoutputs(traindata)
    (validationinputs, validationoutputs) = getinputsandoutputs(validationdata)
    (testinputs, testoutputs) = getinputsandoutputs(testdata)

    # Build all the experiments
    numneurons = 7
    numlayers = 0
    learningrate = 0.2
    activation = :tanh

    (
        train_space_df,
        validation_space_df,
        test_space_df,
    ) = train(numlayers, numneurons, 6, 6, 8, learningrate, 5, 50,
        activation,
        traininputs, trainoutputs,
        validationinputs, validationoutputs,
        testinputs, testoutputs)

    # Concat results to their respective dataframes
    traindata = hcat(traindata, train_space_df)
    validationdata = hcat(validationdata, validation_space_df)
    testdata = hcat(testdata, test_space_df)

    # Save the dataframes to csvs
    CSV.write("traindata.csv", traindata)
    CSV.write("validationdata.csv", validationdata)
    CSV.write("testdata.csv", testdata)
end


using DataFrames
using CSV
using Random
using Distributed

function loadnormalized(path::String)::DataFrame
    data = CSV.read(path, DataFrame)
    # Normalize the data by mapping each column to the -1, 1 range
    for column ∈ names(data)
        data[!, column] = ((data[!, column] .- minimum(data[!, column])) ./
                           (maximum(data[!, column]) .- minimum(data[!, column])))
        data[!, column] = 2 * data[!, column] .- 1
    end

    return data
end

function loadbscdata(path::String)::DataFrame
    data = CSV.read(path, DataFrame)

    # fill missing values with 0
    for column ∈ names(data)
        data[!, column] = coalesce.(data[!, column], 0)
    end
    # Get the unique event names
    event_names = unique(data[!, :event_name])

    # Create a dictionary of event_name => one-hot encoded column
    eventcolumns = Dict(
        Symbol(event_name) => map(x -> x ? 1.0 : 0.0, data[!, :event_name]
                                                      .==
                                                      event_name) for event_name in event_names
    )

    # Turn the following columns into log scale
    tolog = [:amount, :shares, :lockedAmount]

    for column ∈ tolog
        data[!, column] = log.(data[!, column] .+ 1)
    end

    # normalize the following columns
    tonormalize = [:gasUsed, :amount, :shares, :lockedAmount, :lockedDuration, :duration]

    for column ∈ tonormalize
        data[!, column] = ((data[!, column] .- minimum(data[!, column])) ./
                           (maximum(data[!, column]) .- minimum(data[!, column])))
        data[!, column] = 2 * data[!, column] .- 1
    end

    # Build a new dataframe with the normalized columns and the one-hot encoded columns
    newcolumns = Dict(
        Symbol(column) => data[!, column] for column ∈ tonormalize
    )

    # Merge the two dictionaries
    newcolumns = merge(newcolumns, eventcolumns)

    println(keys(newcolumns))

    # Create a new dataframe with the new columns
    newdata = DataFrame(newcolumns)

    return newdata
end

function test()
    alldata = loadbscdata("pancake_data.csv")

    Random.seed!(1234)

    # shuffle rows 
    alldata = alldata[randperm(size(alldata, 1)), :]

    # Split into training, validation and test sets
    trainingperc = 0.6
    validationperc = 0.2

    trainingend = floor(Int, trainingperc * size(alldata, 1))
    validationend = floor(Int, (trainingperc + validationperc) * size(alldata, 1))

    traindata = alldata[1:trainingend, :]
    validationdata = alldata[trainingend+1:validationend, :]
    testdata = alldata[validationend+1:end, :]
    println("Training set size: $(size(traindata, 1))")
    println("Validation set size: $(size(validationdata, 1))")
    println("Test set size: $(size(testdata, 1))")

    # first 2 columns are inputs, last two are outputs
    getinputsandoutputs(data) = (data[:, 6:11], data[:, 6:11])

    (traininputs, trainoutputs) = getinputsandoutputs(traindata)
    (validationinputs, validationoutputs) = getinputsandoutputs(validationdata)
    (testinputs, testoutputs) = getinputsandoutputs(testdata)

    # Build all the experiments
    numneurons = 6:8
    numlayers = 0:3
    learningrates = [0.2, 0.5, 0.9]
    activations = [:linear, :sigmoid, :tanh, :relu]

    # Get the tuples
    paramscombinations = collect(Iterators.product(
        numneurons, numlayers, learningrates, activations))

    # Run the experiments in parallel
    @distributed for (neurons, layers, lr, activation) in paramscombinations
        train(layers, neurons, 6, 6, 8, lr, 400, 50,
            activation,
            traininputs, trainoutputs,
            validationinputs, validationoutputs,
            testinputs, testoutputs)
    end
end
