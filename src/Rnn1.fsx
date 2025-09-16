open System

// Aux functions
let tanh (x: float) = Math.Tanh(x)
let dtanh (y: float) = 1.0 - y * y  // derivate, y = tanh(x)

let softmax (x: float[]) =
    let maxVal = Array.max x
    let exps = x |> Array.map (fun v -> exp (v - maxVal))
    let sumExp = Array.sum exps
    exps |> Array.map (fun v -> v / sumExp)

// RNN structure
type RNN =
    {
        Wxh: float[,]   // input -> hidden
        Whh: float[,]   // hidden -> hidden
        Why: float[,]   // hidden -> output
        bh: float[]     // bias hidden
        by: float[]     // bias output
    }

// Random initialization
let rand = Random()

let initRNN inputSize hiddenSize outputSize =
    let randn rows cols =
        Array2D.init rows cols (fun _ _ -> rand.NextDouble() * 2.0 - 1.0)
    {
        Wxh = randn hiddenSize inputSize
        Whh = randn hiddenSize hiddenSize
        Why = randn outputSize hiddenSize
        bh = Array.init hiddenSize (fun _ -> 0.0)
        by = Array.init outputSize (fun _ -> 0.0)
    }

// Forward pass
let forwardRNN (rnn: RNN) (inputs: float[][]) =
    let timeSteps = inputs.Length
    let hiddenSize = rnn.bh.Length

    let hs = Array.init (timeSteps+1) (fun _ -> Array.zeroCreate hiddenSize)
    let ys = Array.zeroCreate<float[]> timeSteps

    for t in 0 .. timeSteps-1 do
        // h_t = tanh(Wxh * x_t + Whh * h_(t-1) + bh)
        let x = inputs.[t]
        let hPrev = hs.[t]
        let h = Array.zeroCreate hiddenSize
        for i in 0 .. hiddenSize-1 do
            let mutable sum = rnn.bh.[i]
            for j in 0 .. x.Length-1 do
                sum <- sum + rnn.Wxh.[i,j] * x.[j]
            for j in 0 .. hiddenSize-1 do
                sum <- sum + rnn.Whh.[i,j] * hPrev.[j]
            h.[i] <- tanh sum
        hs.[t+1] <- h

        // y_t = softmax(Why * h_t + by)
        let yRaw =
            Array.init rnn.by.Length (fun k ->
                let mutable sum = rnn.by.[k]
                for j in 0 .. hiddenSize-1 do
                    sum <- sum + rnn.Why.[k,j] * h.[j]
                sum)
        ys.[t] <- softmax yRaw
    hs, ys

// ====================
// DEMO: Simple sequence
// ====================

// Data: learning to predict next in sequence [0,1,2] cyclically
let inputs =
    [|
        [|1.0; 0.0; 0.0|]  // represents "0"
        [|0.0; 1.0; 0.0|]  // represents "1"
        [|0.0; 0.0; 1.0|]  // represents "2"
    |]

// Initialize RNN: input=3, hidden=5, output=3
let rnn = initRNN 3 5 3

// Forward
let hs, ys = forwardRNN rnn inputs

printfn "Predictions:"
ys |> Array.iteri (fun t y ->
    printfn "Step %d -> %A" t y
)
