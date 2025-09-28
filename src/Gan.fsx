open System

// Sigmoid
let sigmoid x = 1.0 / (1.0 + exp(-x))
let sigmoidDerivative x = x * (1.0 - x)

// Simple perceptron layer
type Layer = {
    Weights: float[]
    Bias: float
}

let rand = Random()

let initLayer inputSize =
    { Weights = Array.init inputSize (fun _ -> rand.NextDouble() * 2.0 - 1.0)
      Bias = rand.NextDouble() * 2.0 - 1.0 }

let forward (layer: Layer) (inputs: float[]) =
    let mutable sum = layer.Bias
    for i in 0 .. inputs.Length - 1 do
        sum <- sum + layer.Weights.[i] * inputs.[i]
    sigmoid sum

let updateLayer (layer: Layer) (delta: float) (inputs: float[]) lr =
    let newWeights =
        Array.mapi (fun i w -> w + lr * delta * inputs.[i]) layer.Weights
    let newBias = layer.Bias + lr * delta
    { Weights = newWeights; Bias = newBias }

// models
type Generator = { Layer: Layer }
let initGenerator noiseDim = { Layer = initLayer noiseDim }

type Discriminator = { Layer: Layer }
let initDiscriminator inputDim = { Layer = initLayer inputDim }

// forward passes
let generatorForward (gen: Generator) (z: float[]) =
    [| forward gen.Layer z |]   // generate a single value 1D array

let discriminatorForward (disc: Discriminator) (x: float[]) =
    forward disc.Layer x        // "Real" probability

// Training
let train (gen: Generator) (disc: Discriminator) epochs lr =
    let rec loop g d epoch =
        if epoch > epochs then g, d
        else
            // 1. Training Discriminator
            let real = rand.NextDouble() * 0.1 + 0.45
            let realInput = [| real |]
            let realPred = discriminatorForward d realInput
            let realError = 1.0 - realPred
            let realDelta = realError * sigmoidDerivative realPred

            let z = [| rand.NextDouble() |]
            let fake = generatorForward g z
            let fakePred = discriminatorForward d fake
            let fakeError = 0.0 - fakePred
            let fakeDelta = fakeError * sigmoidDerivative fakePred

            // New discriminator after real and fake updates
            let d1 : Discriminator = { Layer = updateLayer d.Layer realDelta realInput lr }
            let d2 : Discriminator = { Layer = updateLayer d1.Layer fakeDelta fake lr }

            // 2. Training Generator
            let z2 = [| rand.NextDouble() |]
            let fake2 = generatorForward g z2
            let pred = discriminatorForward d2 fake2
            let errorG = 1.0 - pred
            let deltaG = errorG * sigmoidDerivative pred

            let newG : Generator = { Layer = updateLayer g.Layer deltaG z2 lr }


            if epoch % 500 = 0 then
                printfn "Epoch %d: Real=%.3f D(real)=%.3f Fake=%.3f D(fake)=%.3f" 
                    epoch real realPred fake.[0] fakePred

            loop newG d2 (epoch+1)

    loop gen disc 1

// Demo 
let gen = initGenerator 1
let disc = initDiscriminator 1

let trainedGen, trainedDisc = train gen disc 5000 0.1

// Test: Generate samples
printfn "Generated samples after training:"
for i in 1 .. 5 do
    let z = [| rand.NextDouble() |]
    let fake = generatorForward trainedGen z
    printfn "Generated sample: %.3f" fake.[0]
