open System

// Aux functions
let tanh (x: float) = Math.Tanh(x)
let dtanh (y: float) = 1.0 - y * y
let softmax (x: float[]) =
    let maxVal = Array.max x
    let exps = x |> Array.map (fun v -> exp (v - maxVal))
    let sumExp = Array.sum exps
    exps |> Array.map (fun v -> v / sumExp)

// One-hot for 0,1,2
let oneHot size idx =
    Array.init size (fun i -> if i = idx then 1.0 else 0.0)

// RNN definition
type RNN =
    {
        Wxh: float[,]
        Whh: float[,]
        Why: float[,]
        bh: float[]
        by: float[]
    }

let rand = Random()
let randn rows cols =
    Array2D.init rows cols (fun _ _ -> rand.NextDouble() * 2.0 - 1.0)

let initRNN inputSize hiddenSize outputSize =
    {
        Wxh = randn hiddenSize inputSize
        Whh = randn hiddenSize hiddenSize
        Why = randn outputSize hiddenSize
        bh = Array.zeroCreate hiddenSize
        by = Array.zeroCreate outputSize
    }

// Forward pass (returns all hidden states and outputs)
let forward (rnn: RNN) (inputs: float[][]) =
    let T = inputs.Length
    let H = rnn.bh.Length
    let hs = Array.init (T+1) (fun _ -> Array.zeroCreate H)
    let ys = Array.zeroCreate<float[]> T
    for t in 0 .. T-1 do
        let x = inputs.[t]
        let hPrev = hs.[t]
        let h = Array.zeroCreate H
        for i in 0 .. H-1 do
            let mutable sum = rnn.bh.[i]
            for j in 0 .. x.Length-1 do sum <- sum + rnn.Wxh.[i,j] * x.[j]
            for j in 0 .. H-1 do sum <- sum + rnn.Whh.[i,j] * hPrev.[j]
            h.[i] <- tanh sum
        hs.[t+1] <- h
        let yRaw =
            Array.init rnn.by.Length (fun k ->
                let mutable sum = rnn.by.[k]
                for j in 0 .. H-1 do sum <- sum + rnn.Why.[k,j] * h.[j]
                sum)
        ys.[t] <- softmax yRaw
    hs, ys

// BPTT training
let train (rnn: RNN) (seq: int[]) inputSize hiddenSize outputSize epochs learningRate =
    let mutable net = rnn
    for epoch in 1 .. epochs do
        // Data preparation
        let inputs = seq |> Array.map (fun i -> oneHot inputSize i)
        let targets = seq |> Array.map (fun i -> oneHot outputSize ((i+1) % outputSize)) // siguiente número

        // Forward
        let hs, ys = forward net inputs

        // Gradient initialization
        let dWxh = Array2D.zeroCreate hiddenSize inputSize
        let dWhh = Array2D.zeroCreate hiddenSize hiddenSize
        let dWhy = Array2D.zeroCreate outputSize hiddenSize
        let dbh = Array.zeroCreate hiddenSize
        let dby = Array.zeroCreate outputSize

        // Backpropagation through time
        let mutable dhNext = Array.zeroCreate hiddenSize
        for t in [inputs.Length-1 .. -1 .. 0] do
            // Output error (softmax + cross-entropy)
            let dy = Array.copy ys.[t]
            for k in 0 .. outputSize-1 do dy.[k] <- dy.[k] - targets.[t].[k]

            // Gradients Why, by
            for k in 0 .. outputSize-1 do
                for j in 0 .. hiddenSize-1 do
                    dWhy.[k,j] <- dWhy.[k,j] + dy.[k] * hs.[t+1].[j]
                dby.[k] <- dby.[k] + dy.[k]

            // Hidden state error
            let dh = Array.zeroCreate hiddenSize
            for j in 0 .. hiddenSize-1 do
                for k in 0 .. outputSize-1 do
                    dh.[j] <- dh.[j] + net.Why.[k,j] * dy.[k]
                dh.[j] <- dh.[j] + dhNext.[j]
                dh.[j] <- dh.[j] * dtanh(hs.[t+1].[j])

            // Gradients Wxh, Whh, bh
            for j in 0 .. hiddenSize-1 do
                for i in 0 .. inputSize-1 do
                    dWxh.[j,i] <- dWxh.[j,i] + dh.[j] * inputs.[t].[i]
                for i in 0 .. hiddenSize-1 do
                    dWhh.[j,i] <- dWhh.[j,i] + dh.[j] * hs.[t].[i]
                dbh.[j] <- dbh.[j] + dh.[j]

            dhNext <- dh

        // Update weights and biases
        for i in 0 .. hiddenSize-1 do
            for j in 0 .. inputSize-1 do net.Wxh.[i,j] <- net.Wxh.[i,j] - learningRate * dWxh.[i,j]
            for j in 0 .. hiddenSize-1 do net.Whh.[i,j] <- net.Whh.[i,j] - learningRate * dWhh.[i,j]
            net.bh.[i] <- net.bh.[i] - learningRate * dbh.[i]

        for k in 0 .. outputSize-1 do
            for j in 0 .. hiddenSize-1 do net.Why.[k,j] <- net.Why.[k,j] - learningRate * dWhy.[k,j]
            net.by.[k] <- net.by.[k] - learningRate * dby.[k]

        if epoch % 100 = 0 then
            let _, preds = forward net inputs
            let loss =
                [| for t in 0 .. preds.Length-1 ->
                    -log(preds.[t].[ seq.[ (t+1) % seq.Length ] ]) |]
                |> Array.sum

            printfn "Epoch %d, Loss: %.4f" epoch loss
    net

// ====================
// DEMO: Simple sequence
// ====================

// Simple sequence: 0 -> 1 -> 2 -> 0 -> ...
let seq = [|0;1;2|]
let inputSize, hiddenSize, outputSize = 3, 5, 3

let rnn = initRNN inputSize hiddenSize outputSize
let trained = train rnn seq inputSize hiddenSize outputSize 1000 0.1

// Predictions after training
let inputs = seq |> Array.map (fun i -> oneHot inputSize i)
let _, preds = forward trained inputs
printfn "\nPredictions as a vectos:"
preds |> Array.iteri (fun i p ->
    printfn "Input %d -> %A" seq.[i] p
)

// Aux: argmax function
let argmax (arr: float[]) =
    arr |> Array.mapi (fun i v -> i, v) |> Array.maxBy snd |> fst

printfn "\nPredictions as a argmax:"
preds |> Array.iteri (fun i p ->
    let predIdx = argmax p
    printfn "Input %d -> Prediction: %d (Prob: %.2f)" seq.[i] predIdx p.[predIdx]
)

