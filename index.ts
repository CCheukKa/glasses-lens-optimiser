import { NeuralNetwork } from "./lib/NeuralNetwork";

const nn = new NeuralNetwork([1, 3, 5, 3, 1]);
console.log(nn.predict([0.5]));
// nn.logDebugInfo();