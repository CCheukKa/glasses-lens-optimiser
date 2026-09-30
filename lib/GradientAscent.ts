import { MathExtra } from "./MathExtra";
import { NeuralNetwork } from "./NeuralNetwork";
import { PathType, RayTracer, Scene } from "./RayTracer";

export enum OptimiserType {
    Adam = "adam",
    Rmsprop = "rmsprop",
}
export namespace GradientAscent {
    export type TrainingProgress = {
        iteration: number;
        loss: number;
    };

    export type AdaptiveOptimiser = {
        update(
            neuralNetwork: NeuralNetwork,
            gradient: NeuralNetwork,
            learningRate: number,
        ): void;
    };

    export function evaluateLoss(neuralNetwork: NeuralNetwork, scene: Scene): number {
        scene.setLensFunction((x: number) => neuralNetwork.predict([x])[0]);
        return RayTracer.traceScene(scene).map(ray => {
            const lastSegment = ray[ray.length - 1];
            if (lastSegment.type !== PathType.Outgoing) {
                throw new Error("Last segment of ray is not Outgoing");
            }
            return Math.pow(MathExtra.distanceBetweenRayAndPoint(lastSegment, scene.target.position), 2);
        }).reduce((a, b) => a + b, 0);
    }

    export function computeGradient(neuralNetwork: NeuralNetwork, scene: Scene, delta: number = 1e-5): NeuralNetwork {
        const originalLoss = evaluateLoss(neuralNetwork, scene);
        const gradient = neuralNetwork.clone();

        gradient.layers.forEach(layer => {
            layer.nodes.forEach(node => {
                node.weights.fill(0);
                node.bias = 0;
            });
        });

        for (let i = 0; i < neuralNetwork.layers.length; i++) {
            for (let j = 0; j < neuralNetwork.layers[i].nodes.length; j++) {
                const node = neuralNetwork.layers[i].nodes[j];
                const gradientNode = gradient.layers[i].nodes[j];

                for (let k = 0; k < node.weights.length; k++) {
                    const originalWeight = node.weights[k]!;
                    node.weights[k] = originalWeight + delta;
                    gradientNode.weights[k] = (evaluateLoss(neuralNetwork, scene) - originalLoss) / delta;
                    node.weights[k] = originalWeight;
                }

                const originalBias = node.bias;
                node.bias = originalBias + delta;
                gradientNode.bias = (evaluateLoss(neuralNetwork, scene) - originalLoss) / delta;
                node.bias = originalBias;
            }
        }
        return gradient;
    }

    class RmspropOptimiser implements AdaptiveOptimiser {
        private readonly squaredGradient: NeuralNetwork;

        constructor(
            neuralNetwork: NeuralNetwork,
            private readonly decayRate: number = 0.9,
            private readonly epsilon: number = 1e-8,
        ) {
            this.squaredGradient = neuralNetwork.clone();
            this.resetState();
        }

        update(neuralNetwork: NeuralNetwork, gradient: NeuralNetwork, learningRate: number): void {
            forEachParameter(neuralNetwork, gradient, this.squaredGradient, (parameter, gradientValue, state) => {
                state.value = this.decayRate * state.value + (1 - this.decayRate) * gradientValue * gradientValue;
                return parameter - learningRate * gradientValue / (Math.sqrt(state.value) + this.epsilon);
            });
        }

        private resetState(): void {
            this.squaredGradient.layers.forEach(layer => {
                layer.nodes.forEach(node => {
                    node.weights.fill(0);
                    node.bias = 0;
                });
            });
        }
    }

    class AdamOptimiser implements AdaptiveOptimiser {
        private readonly firstMoment: NeuralNetwork;
        private readonly secondMoment: NeuralNetwork;
        private iteration = 0;

        constructor(
            neuralNetwork: NeuralNetwork,
            private readonly beta1: number = 0.9,
            private readonly beta2: number = 0.999,
            private readonly epsilon: number = 1e-8,
        ) {
            this.firstMoment = neuralNetwork.clone();
            this.secondMoment = neuralNetwork.clone();
            this.resetState();
        }

        update(neuralNetwork: NeuralNetwork, gradient: NeuralNetwork, learningRate: number): void {
            this.iteration++;
            const firstMomentCorrection = 1 - Math.pow(this.beta1, this.iteration);
            const secondMomentCorrection = 1 - Math.pow(this.beta2, this.iteration);

            forEachParameter(neuralNetwork, gradient, this.firstMoment, (parameter, gradientValue, firstState, layerIndex, nodeIndex, parameterIndex, isBias) => {
                const secondState = isBias
                    ? { value: this.secondMoment.layers[layerIndex]!.nodes[nodeIndex]!.bias }
                    : { value: this.secondMoment.layers[layerIndex]!.nodes[nodeIndex]!.weights[parameterIndex]! };

                firstState.value = this.beta1 * firstState.value + (1 - this.beta1) * gradientValue;
                secondState.value = this.beta2 * secondState.value + (1 - this.beta2) * gradientValue * gradientValue;

                if (isBias) {
                    this.secondMoment.layers[layerIndex]!.nodes[nodeIndex]!.bias = secondState.value;
                } else {
                    this.secondMoment.layers[layerIndex]!.nodes[nodeIndex]!.weights[parameterIndex] = secondState.value;
                }

                const correctedFirstMoment = firstState.value / firstMomentCorrection;
                const correctedSecondMoment = secondState.value / secondMomentCorrection;
                return parameter - learningRate * correctedFirstMoment / (Math.sqrt(correctedSecondMoment) + this.epsilon);
            });
        }

        private resetState(): void {
            for (const network of [this.firstMoment, this.secondMoment]) {
                network.layers.forEach(layer => {
                    layer.nodes.forEach(node => {
                        node.weights.fill(0);
                        node.bias = 0;
                    });
                });
            }
        }
    }

    export function createOptimiser(
        type: OptimiserType,
        neuralNetwork: NeuralNetwork,
    ): AdaptiveOptimiser {
        return type === OptimiserType.Adam
            ? new AdamOptimiser(neuralNetwork)
            : new RmspropOptimiser(neuralNetwork);
    }

    function forEachParameter(
        neuralNetwork: NeuralNetwork,
        gradient: NeuralNetwork,
        state: NeuralNetwork,
        update: (
            parameter: number,
            gradient: number,
            state: { value: number },
            layerIndex: number,
            nodeIndex: number,
            parameterIndex: number,
            isBias: boolean,
        ) => number,
    ): void {
        neuralNetwork.layers.forEach((layer, layerIndex) => {
            layer.nodes.forEach((node, nodeIndex) => {
                const gradientNode = gradient.layers[layerIndex]!.nodes[nodeIndex]!;
                const stateNode = state.layers[layerIndex]!.nodes[nodeIndex]!;

                node.weights = node.weights.map((parameter, parameterIndex) => {
                    const stateValue = { value: stateNode.weights[parameterIndex]! };
                    const updatedParameter = update(
                        parameter,
                        gradientNode.weights[parameterIndex]!,
                        stateValue,
                        layerIndex,
                        nodeIndex,
                        parameterIndex,
                        false,
                    );
                    stateNode.weights[parameterIndex] = stateValue.value;
                    return updatedParameter;
                });

                const biasState = { value: stateNode.bias };
                node.bias = update(
                    node.bias,
                    gradientNode.bias,
                    biasState,
                    layerIndex,
                    nodeIndex,
                    0,
                    true,
                );
                stateNode.bias = biasState.value;
            });
        });
    }

    export function gradientAscent(
        neuralNetwork: NeuralNetwork,
        scene: Scene,
        learningRate: number = 0.01,
        iterations: number = 100,
        optimiser: AdaptiveOptimiser = createOptimiser(OptimiserType.Adam, neuralNetwork),
    ): NeuralNetwork {
        for (let iter = 0; iter < iterations; iter++) {
            const gradient = computeGradient(neuralNetwork, scene);
            optimiser.update(neuralNetwork, gradient, learningRate);
        }
        return neuralNetwork;
    }

    export async function* gradientAscentAsync(
        neuralNetwork: NeuralNetwork,
        scene: Scene,
        learningRate: number = 0.01,
        iterations: number = 100,
        optimiser: AdaptiveOptimiser = createOptimiser(OptimiserType.Adam, neuralNetwork),
    ): AsyncGenerator<TrainingProgress, NeuralNetwork, void> {
        for (let iter = 0; iter < iterations; iter++) {
            const gradient = computeGradient(neuralNetwork, scene);
            optimiser.update(neuralNetwork, gradient, learningRate);

            yield {
                iteration: iter + 1,
                loss: evaluateLoss(neuralNetwork, scene),
            };
        }

        return neuralNetwork;
    }
}