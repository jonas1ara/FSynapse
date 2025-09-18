open System

// Sigmoid and its derivative
let sigmoid x = 1.0 / (1.0 + exp(-x))
let sigmoidDerivative x = x * (1.0 - x)

// Autoencoder type (1 hidden layer)
type Autoencoder = {
    EncoderWeights: float[,]
    EncoderBiases: float[]
    DecoderWeights: float[,]
    DecoderBiases: float[]
}

// Random initialization
let rand = Random()

let initAutoencoder inputSize hiddenSize =
    {
        EncoderWeights = Array2D.init hiddenSize inputSize (fun _ _ -> rand.NextDouble() * 2.0 - 1.0)
        EncoderBiases  = Array.init hiddenSize (fun _ -> rand.NextDouble() * 2.0 - 1.0)
        DecoderWeights = Array2D.init inputSize hiddenSize (fun _ _ -> rand.NextDouble() * 2.0 - 1.0)
        DecoderBiases  = Array.init inputSize (fun _ -> rand.NextDouble() * 2.0 - 1.0)
    }

// Forward pass for one layer
let layerForward (weights: float[,]) (biases: float[]) (inputs: float[]) =
    Array.init (weights.GetLength 0) (fun i ->
        let mutable sum = biases.[i]
        for j in 0 .. weights.GetLength 1 - 1 do
            sum <- sum + weights.[i,j] * inputs.[j]
        sigmoid sum
    )

// Forward complete: encoder -> decoder
let forward (ae: Autoencoder) (inputs: float[]) =
    let hidden = layerForward ae.EncoderWeights ae.EncoderBiases inputs
    let reconstructed = layerForward ae.DecoderWeights ae.DecoderBiases hidden
    hidden, reconstructed

// Backpropagation
let train (ae: Autoencoder) (dataset: float[][]) epochs lr =
    let mutable net = ae
    for epoch in 1 .. epochs do
        let mutable epochLoss = 0.0
        for x in dataset do
            // Forward
            let hidden = layerForward net.EncoderWeights net.EncoderBiases x
            let output = layerForward net.DecoderWeights net.DecoderBiases hidden

            // Error
            let errors = Array.map2 (fun t o -> t - o) x output
            epochLoss <- epochLoss + (errors |> Array.map (fun e -> e*e) |> Array.sum)

            // Deltas output
            let outputDeltas = Array.map2 (fun err o -> err * sigmoidDerivative o) errors output

            // Deltas hidden
            let hiddenErrors =
                Array.init hidden.Length (fun j ->
                    [for k in 0 .. outputDeltas.Length-1 -> outputDeltas.[k] * net.DecoderWeights.[k,j]]
                    |> List.sum
                )
            let hiddenDeltas = Array.map2 (fun err h -> err * sigmoidDerivative h) hiddenErrors hidden

            // Update weights decoder
            for i in 0 .. net.DecoderWeights.GetLength 0 - 1 do
                for j in 0 .. net.DecoderWeights.GetLength 1 - 1 do
                    net.DecoderWeights.[i,j] <- net.DecoderWeights.[i,j] + lr * outputDeltas.[i] * hidden.[j]
                net.DecoderBiases.[i] <- net.DecoderBiases.[i] + lr * outputDeltas.[i]

            // Update weights encoder
            for i in 0 .. net.EncoderWeights.GetLength 0 - 1 do
                for j in 0 .. net.EncoderWeights.GetLength 1 - 1 do
                    net.EncoderWeights.[i,j] <- net.EncoderWeights.[i,j] + lr * hiddenDeltas.[i] * x.[j]
                net.EncoderBiases.[i] <- net.EncoderBiases.[i] + lr * hiddenDeltas.[i]

        if epoch % 100 = 0 then
            printfn "Epoch %d - Loss: %.4f" epoch (epochLoss / float dataset.Length)
    net

// ===================
// Demo: 4D identity
// ===================
let trainingData =
    [| [|1.0; 0.0; 0.0; 0.0|]
       [|0.0; 1.0; 0.0; 0.0|]
       [|0.0; 0.0; 1.0; 0.0|]
       [|0.0; 0.0; 0.0; 1.0|] |]

let autoencoder = initAutoencoder 4 2
let trained = train autoencoder trainingData 1000 0.5

// Test reconstruction
for sample in trainingData do
    let _, reconstructed = forward trained sample
    printfn "Input: %A -> reconstructed: %A" sample reconstructed
