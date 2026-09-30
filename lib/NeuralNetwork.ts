class Node {
    public weights: number[];
    public bias: number;

    constructor(numInputs: number) {
        this.weights = Array.from({ length: numInputs }, () => Math.random() * 2 - 1);
        this.bias = numInputs === 0 ? 0 : Math.random() * 2 - 1;
    }
}
class Layer {
    public nodes: Node[];

    constructor(numInputs: number, numNodes: number) {
        this.nodes = Array.from({ length: numNodes }, () => new Node(numInputs));
    }
}
export class NeuralNetwork {
    public activationFunction: (x: number) => number;
    public inputNodes: number;
    public layers: Layer[];

    constructor(numNodesInLayer: number[], activationFunction: (x: number) => number) {
        this.activationFunction = activationFunction;
        this.inputNodes = numNodesInLayer[0]!;
        this.layers = numNodesInLayer.slice(1).map((numNodes, index) => {
            const numInputs = index === 0 ? this.inputNodes : numNodesInLayer[index]!;
            return new Layer(numInputs, numNodes);
        });
    }

    predict(inputs: number[]): number[] {
        if (inputs.length !== this.inputNodes) {
            throw new Error(`Expected ${this.inputNodes} inputs, but got ${inputs.length}`);
        }

        let output = inputs;
        for (const layer of this.layers) {
            output = layer.nodes.map(node => {
                const weightedSum = node.weights.reduce((sum, weight, index) => sum + weight * output[index]!, 0);
                return this.activationFunction(weightedSum + node.bias);
            });
        }

        return output;
    }
    clone(): NeuralNetwork {
        const newNetwork = new NeuralNetwork([], this.activationFunction);
        newNetwork.inputNodes = this.inputNodes;
        newNetwork.layers = this.layers.map(layer =>
            new Layer(layer.nodes[0]!.weights.length, layer.nodes.length)
        );
        newNetwork.layers.forEach((newLayer, index) => {
            newLayer.nodes.forEach((newNode, nodeIndex) => {
                const oldNode = this.layers[index]!.nodes[nodeIndex]!;
                newNode.weights = [...oldNode.weights];
                newNode.bias = oldNode.bias;
            });
        });
        return newNetwork;
    }
    logDebugInfo(verbose: boolean = false) {
        let debugInfo: string[] = [];
        debugInfo.push(`Neural Network Debug Info:`);
        debugInfo.push(`Activation Function: ${this.activationFunction.toString()}`);
        debugInfo.push(`Layers: [${this.inputNodes}] -> ${this.layers.map((layer, index) => index === this.layers.length - 1 ? `[${layer.nodes.length}]` : layer.nodes.length).join(" -> ")}`);
        if (verbose) {
            this.layers.forEach((layer, index) => {
                const lastLayer = index === this.layers.length - 1;
                debugInfo.push("┃");
                debugInfo.push(`${lastLayer ? "┗" : "┣"}━ Layer ${index}: ${layer.nodes.length} nodes`);
                layer.nodes.forEach((node, nodeIndex) => {
                    const lastNode = nodeIndex === layer.nodes.length - 1;
                    debugInfo.push(`${lastLayer ? " " : "┃"}   ${lastNode ? "└" : "├"}─ Node ${nodeIndex}:`);
                    debugInfo.push(`${lastLayer ? " " : "┃"}   ${lastNode ? " " : "│"}   ├─ Weights: [${node.weights.map(w => w.toFixed(4)).join(", ")}]`);
                    debugInfo.push(`${lastLayer ? " " : "┃"}   ${lastNode ? " " : "│"}   └─ Bias   : ${node.bias.toFixed(4)}`);
                });
            });
        }
        console.log(debugInfo.join("\n"));
    }
}