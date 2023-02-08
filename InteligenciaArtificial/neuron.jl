

struct NeuronForwardData
    inputs::Array{Float64}
    localfield::Float64
    output::Float64
end

"""
    forward(weights, inputs)

Computes the forward pass of the network and returns the relevant results.
"""
function forward(
    weights::Array{Float64},
    inputs::Array{Float64},
    ϕ::Function
)::NeuronForwardData

    localfield = sum([weight * input for (weight, input) ∈ zip(weights, inputs)])

    output = ϕ(localfield)

    return NeuronForwardData(inputs, localfield, output)
end

"""
    ∇(data, target, diffL, diffϕ)

Computes the gradient of the weights. L' is the derivative of the loss, and ϕ' is the
derivative of the activation function.
"""
function ∇(
    data::NeuronForwardData,
    target::Float64,
    diffL::Function,
    diffϕ::Function
)::Array{Float64}
    return [
        diffL(target, data.output) * diffϕ(data.localfield) * input
        for input ∈ data.inputs
    ]
end

function test()
    all_inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    all_outputs = [
        0.0,
        0.0,
        0.0,
        1.0
    ]

    L = (target, output) -> ((target - output)^2) / 2
    diffL = (target, output) -> -(target - output)
    ϕ = localfield -> 1 / (1 + exp(-localfield))
    diffϕ = localfield -> ϕ(localfield) * (1 - ϕ(localfield))

    lr = 1 * 10^-2

    weights = rand(Float64, 2) * 2 .- 1
    for _ in 1:100000
        gradients = []
        losses = []
        for (inputs, target) ∈ zip(all_inputs, all_outputs)
            result = forward(weights, inputs, ϕ)
            loss = L(target, result.output)
            gradient = ∇(result, target, diffL, diffϕ)
            push!(gradients, gradient)
            push!(losses, loss)
        end

        meanloss = sum(losses) / length(losses)
        meangrad = [
            sum([gradient[i] for gradient ∈ gradients]) / length(gradients)
            for i ∈ 1:length(gradients[1])
        ]

        println(meanloss)

        descent = lr .* -meangrad
        weights .+= descent
    end

    println(weights)

    for inputs ∈ all_inputs
        println(inputs)
        output = forward(weights, inputs, ϕ)
        println(output.output)
    end
end