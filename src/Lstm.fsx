open System

// Activations
let sigmoid x = 1.0 / (1.0 + exp(-x))
let tanh' x = Math.Tanh x

let rand = Random()

// weights and biases initialization
let initWeights input hidden =
    Array.init hidden (fun _ ->
        Array.init (input + hidden) (fun _ -> rand.NextDouble() * 2.0 - 1.0))

let initBias hidden =
    Array.init hidden (fun _ -> rand.NextDouble() * 2.0 - 1.0)

// LSTM Cell Definition
type LSTM = {
    Wf: float[][]
    Wi: float[][]
    Wo: float[][]
    Wc: float[][]
    bf: float[]
    bi: float[]
    bo: float[]
    bc: float[]
    hiddenSize: int
}

// Initialize LSTM
let initLSTM inputSize hiddenSize =
    {
        Wf = initWeights inputSize hiddenSize
        Wi = initWeights inputSize hiddenSize
        Wo = initWeights inputSize hiddenSize
        Wc = initWeights inputSize hiddenSize
        bf = initBias hiddenSize
        bi = initBias hiddenSize
        bo = initBias hiddenSize
        bc = initBias hiddenSize
        hiddenSize = hiddenSize
    }

// Forward pass
let lstmForward (lstm: LSTM) (inputs: float[][]) =
    let hiddenSize = lstm.hiddenSize
    let mutable h = Array.zeroCreate hiddenSize
    let mutable c = Array.zeroCreate hiddenSize

    let concat hPrev x =
        Array.append hPrev x

    for x in inputs do
        let combined = concat h x
        let dot (W: float[][]) (v: float[]) (b: float[]) =
            Array.mapi (fun i row ->
                Array.map2 (*) row v |> Array.sum |> fun s -> s + b.[i]) W

        let f = dot lstm.Wf combined lstm.bf |> Array.map sigmoid
        let i = dot lstm.Wi combined lstm.bi |> Array.map sigmoid
        let o = dot lstm.Wo combined lstm.bo |> Array.map sigmoid
        let cTilde = dot lstm.Wc combined lstm.bc |> Array.map tanh'

        c <- Array.map3 (fun f i cT ->
            f * c.[Array.IndexOf(c, c.[0])] + i * cT) f i cTilde
        h <- Array.map2 (fun o ct -> o * tanh' ct) o c

    h, c

// Demo
let lstm = initLSTM 3 2

// Input sequence (4 time steps, input size 3)
let inputs =
    [|
        [| 0.1; 0.2; 0.3 |]
        [| 0.4; 0.3; 0.2 |]
        [| 0.2; 0.5; 0.1 |]
        [| 0.3; 0.2; 0.4 |]
    |]

// Forward pass through the LSTM
let (hFinal, cFinal) = lstmForward lstm inputs

printfn "Final hidden state (h): %A" hFinal
printfn "Final cell state (c): %A" cFinal
