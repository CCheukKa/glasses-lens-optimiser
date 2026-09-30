import { MathExtra } from "./MathExtra";
import { NeuralNetwork } from "./NeuralNetwork";
import { PathType, RayTracer, Scene } from "./RayTracer";

export namespace GradientAscent {
    export function evaluateLoss(neuralNetwork: NeuralNetwork, scene: Scene): number {
        scene.setLensFunction((x: number) => neuralNetwork.predict([x])[0]);
        return RayTracer.traceScene(scene).map(ray => {
            const lastSegment = ray[ray.length - 1];
            if (lastSegment.type !== PathType.Outgoing) {
                throw new Error("Last segment of ray is not Outgoing");
            }
            return MathExtra.distanceBetweenRayAndPoint(lastSegment, scene.target.position);
        }).reduce((a, b) => a + b, 0);
    }

    export function computeGradient(neuralNetwork: NeuralNetwork, scene: Scene, delta: number = 1e-5): NeuralNetwork {
        const originalLoss = evaluateLoss(neuralNetwork, scene);
        const gradient = neuralNetwork.clone();
        for (let i = 0; i < neuralNetwork.layers.length; i++) {
            for (let j = 0; j < neuralNetwork.layers[i].nodes.length; j++) {
                const originalWeight = neuralNetwork.layers[i].nodes[j].weights[0];
                neuralNetwork.layers[i].nodes[j].weights[0] += delta;
                const newLoss = evaluateLoss(neuralNetwork, scene);
                gradient.layers[i].nodes[j].weights[0] = (newLoss - originalLoss) / delta;
                neuralNetwork.layers[i].nodes[j].weights[0] = originalWeight; // Restore original weight
            }
        }
        return gradient;
    }

    export function gradientAscent(neuralNetwork: NeuralNetwork, scene: Scene, learningRate: number = 0.01, iterations: number = 100): NeuralNetwork {
        for (let iter = 0; iter < iterations; iter++) {
            const gradient = computeGradient(neuralNetwork, scene);
            for (let i = 0; i < neuralNetwork.layers.length; i++) {
                for (let j = 0; j < neuralNetwork.layers[i].nodes.length; j++) {
                    neuralNetwork.layers[i].nodes[j].weights[0] -= learningRate * gradient.layers[i].nodes[j].weights[0];
                }
            }
        }
        return neuralNetwork;
    }
}