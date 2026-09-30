import { NeuralNetwork } from "./lib/NeuralNetwork";
import { Scene } from "./lib/RayTracer";
import { Vector2 } from "./lib/Vector";

const neuralNetwork = new NeuralNetwork([1, 3, 5, 3, 1]);

const lightSourceXStart = -1;
const lightSourceXEnd = 1;
const lightSourceCount = 20;
const lightSources = Array.from({ length: lightSourceCount }, (_, i) => {
    const x = lightSourceXStart + (i / (lightSourceCount - 1)) * (lightSourceXEnd - lightSourceXStart);
    return { position: new Vector2(x, 1), direction: new Vector2(0, -1) };
});
const lens = {
    position: new Vector2(0, 0),
    refractiveIndex: 1.5,
    function: (x: number) => neuralNetwork.predict([x])[0]
};
const target = { position: new Vector2(0, -1) };
const scene = new Scene(lightSources, lens, target);