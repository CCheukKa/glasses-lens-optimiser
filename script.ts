import { NeuralNetwork } from "./lib/NeuralNetwork";
import { PathType, RayTracer, Scene } from "./lib/RayTracer";
import { Vector2 } from "./lib/MathExtra";
import { GradientAscent } from "./lib/GradientAscent";

const LEARNING_RATE = 0.001;
const MAX_ITERATIONS = 1000000;
const relu = (x: number): number => Math.max(0, x);
const leakyRelu = (x: number): number => (x > 0 ? x : 0.01 * x);
const tanh = (x: number): number => Math.tanh(x);
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const neuralNetwork = new NeuralNetwork([1, 3, 5, 7, 5, 3, 1], tanh);

const lightSourceXStart = -1;
const lightSourceXEnd = 1;
const lightSourceCount = 20;
const lightSources = Array.from({ length: lightSourceCount }, (_, i) => {
    const x = lightSourceXStart + (i / (lightSourceCount - 1)) * (lightSourceXEnd - lightSourceXStart);
    return { position: new Vector2(x, 2.5), direction: new Vector2(0, -1) };
});
const lens = {
    position: new Vector2(0, 1),
    refractiveIndex: 1.5,
    function: (x: number) => neuralNetwork.predict([x])[0],
};
const target = { position: new Vector2(0, -2.5) };
const scene = new Scene(lightSources, lens, target);

async function waitForNextFrame(): Promise<void> {
    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
}

train();

async function train(): Promise<void> {
    drawScene(scene);
    console.log("Initial loss:", GradientAscent.evaluateLoss(neuralNetwork, scene));
    await waitForNextFrame();

    for await (const progress of GradientAscent.gradientAscentAsync(neuralNetwork, scene, LEARNING_RATE, MAX_ITERATIONS)) {
        drawScene(scene);
        if (progress.iteration % 10 === 0 || progress.iteration === 1) {
            console.log(`Iteration ${progress.iteration}, loss: ${progress.loss}`);
        }
        await waitForNextFrame();
    }

    console.log("Training complete. Final loss:", GradientAscent.evaluateLoss(neuralNetwork, scene));
}

function drawScene(scene: Scene) {
    const canvas = document.getElementById("myCanvas") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        console.error("Failed to get canvas context");
        return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // draw light sources
    const lightSourceRadius = 2;
    scene.lightSources.forEach(lightSource => {
        ctx.beginPath();
        ctx.arc(lightSource.position.x * 100 + canvas.width / 2, -lightSource.position.y * 100 + canvas.height / 2, lightSourceRadius, 0, 2 * Math.PI);
        ctx.fillStyle = "yellow";
        ctx.fill();
    });

    // draw lens
    const surfaceThickness = 2;
    const normalLength = 0.4;
    const normalThickness = 1;
    scene.discretisedLens.forEach(surface => {
        // draw surface
        ctx.beginPath();
        ctx.moveTo(surface.start.x * 100 + canvas.width / 2, -surface.start.y * 100 + canvas.height / 2);
        ctx.lineTo(surface.end.x * 100 + canvas.width / 2, -surface.end.y * 100 + canvas.height / 2);
        ctx.strokeStyle = "cyan";
        ctx.lineWidth = surfaceThickness;
        ctx.stroke();

        // draw normal at center of surface
        const center = surface.start.add(surface.end).scale(0.5);
        const normalEnd = center.add(surface.normal.scale(normalLength));
        ctx.beginPath();
        ctx.moveTo(center.x * 100 + canvas.width / 2, -center.y * 100 + canvas.height / 2);
        ctx.lineTo(normalEnd.x * 100 + canvas.width / 2, -normalEnd.y * 100 + canvas.height / 2);
        ctx.strokeStyle = "magenta";
        ctx.lineWidth = normalThickness;
        ctx.stroke();
    });

    // draw target
    const targetRadius = 5;
    ctx.beginPath();
    ctx.arc(scene.target.position.x * 100 + canvas.width / 2, -scene.target.position.y * 100 + canvas.height / 2, targetRadius, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();

    // draw rays
    const rays = RayTracer.traceScene(scene);
    rays.forEach(ray => {
        ctx.beginPath();
        ray.forEach(segment => {
            if (segment.type === PathType.Intersection) {
                ctx.moveTo(segment.start.x * 100 + canvas.width / 2, -segment.start.y * 100 + canvas.height / 2);
                ctx.lineTo(segment.end.x * 100 + canvas.width / 2, -segment.end.y * 100 + canvas.height / 2);
            } else if (segment.type === PathType.Outgoing) {
                ctx.moveTo(segment.start.x * 100 + canvas.width / 2, -segment.start.y * 100 + canvas.height / 2);
                ctx.lineTo(
                    (segment.start.x + segment.direction.x * 2) * 500 + canvas.width / 2,
                    -(segment.start.y + segment.direction.y * 2) * 500 + canvas.height / 2,
                );
            }
        });
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}