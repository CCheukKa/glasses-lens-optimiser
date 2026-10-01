import { NeuralNetwork } from "./lib/NeuralNetwork";
import { PathType, RayTracer, Scene } from "./lib/RayTracer";
import { MathExtra, Vector2 } from "./lib/MathExtra";
import { GradientAscent, OptimiserType } from "./lib/GradientAscent";

const LEARNING_RATE = 0.005;
const MAX_ITERATIONS = 5000;
const AUTO_STOP_PATIENCE = 500;
const LOSS_IMPROVEMENT_TOLERANCE = 1e-5;
const OPTIMISER = OptimiserType.Adam;
const WORLD_TO_CANVAS_SCALE = 100;
const OUTGOING_RAY_LENGTH = 5;
const relu = (x: number): number => Math.max(0, x);
const leakyRelu = (x: number): number => (x > 0 ? x : 0.01 * x);
const tanh = (x: number): number => Math.tanh(x);
const tanh2 = (x: number): number => Math.tanh(x) * 2;
const sigmoid2 = (x: number): number => (1 / (1 + Math.exp(-x)) - 0.5) * 8;
const a = (x: number): number => Math.sqrt(Math.abs(x)) * Math.tanh(x);
const b = (x: number): number => MathExtra.clamp(Math.sign(x) * (Math.exp(Math.abs(x)) - 1), -10, 10);
const neuralNetwork = new NeuralNetwork([1, 3, 5, 7, 5, 2], tanh, true);

const lightSourceXStart = -2;
const lightSourceXEnd = 2;
const lightSourceCount = 20;
const lightSources = Array.from({ length: lightSourceCount }, (_, i) => {
    const x = lightSourceXStart + (i / (lightSourceCount - 1)) * (lightSourceXEnd - lightSourceXStart);
    return { position: new Vector2(x, 2.5), direction: new Vector2(0, -1) };
});
const lens = {
    position: new Vector2(0, 1),
    refractiveIndex: 2,
    function: (x: number) => neuralNetwork.predict([x]),
};
const target = { position: new Vector2(0, -2.5) };
const scene = new Scene(lightSources, lens, target);

function toCanvasPoint(point: Vector2, canvas: HTMLCanvasElement): Vector2 {
    return new Vector2(
        point.x * WORLD_TO_CANVAS_SCALE + canvas.width / 2,
        -point.y * WORLD_TO_CANVAS_SCALE + canvas.height / 2,
    );
}

async function waitForNextFrame(): Promise<void> {
    await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
}

// drawScene(scene);

train();
async function train(): Promise<void> {
    drawScene(scene);
    const initialLoss = GradientAscent.evaluateLoss(neuralNetwork, scene);
    console.log("Initial loss:", initialLoss);
    await waitForNextFrame();

    const optimiser = GradientAscent.createOptimiser(OPTIMISER, neuralNetwork);
    let bestLoss = initialLoss;
    let stagnantIterations = 0;

    for await (const progress of GradientAscent.gradientAscentAsync(
        neuralNetwork,
        scene,
        LEARNING_RATE,
        MAX_ITERATIONS,
        optimiser,
    )) {
        drawScene(scene);
        if (progress.iteration % 10 === 0 || progress.iteration === 1) {
            console.log(`Iteration ${progress.iteration}, loss: ${progress.loss}`);
        }
        await waitForNextFrame();

        if (!Number.isFinite(progress.loss)) {
            console.warn(`Stopping training: loss became ${progress.loss}.`);
            break;
        }

        if (bestLoss - progress.loss > LOSS_IMPROVEMENT_TOLERANCE) {
            bestLoss = progress.loss;
            stagnantIterations = 0;
        } else {
            stagnantIterations++;
        }

        if (stagnantIterations >= AUTO_STOP_PATIENCE) {
            console.log(
                `Stopping training: no meaningful improvement for ${AUTO_STOP_PATIENCE} iterations.`,
            );
            break;
        }
    }

    console.log("Training complete. Final loss:", bestLoss);
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
        const position = toCanvasPoint(lightSource.position, canvas);
        ctx.beginPath();
        ctx.arc(position.x, position.y, lightSourceRadius, 0, 2 * Math.PI);
        ctx.fillStyle = "yellow";
        ctx.fill();
    });

    // draw lens
    const surfaceThickness = 2;
    const normalLength = 0.4;
    const normalThickness = 1;
    scene.discretisedLens.forEach(surface => {
        const start = toCanvasPoint(surface.start, canvas);
        const end = toCanvasPoint(surface.end, canvas);
        // draw surface
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.strokeStyle = "cyan";
        ctx.lineWidth = surfaceThickness;
        ctx.stroke();

        // draw normal at center of surface
        const center = surface.start.add(surface.end).scale(0.5);
        const normalEnd = center.add(surface.normal.scale(normalLength));
        const canvasCenter = toCanvasPoint(center, canvas);
        const canvasNormalEnd = toCanvasPoint(normalEnd, canvas);
        ctx.beginPath();
        ctx.moveTo(canvasCenter.x, canvasCenter.y);
        ctx.lineTo(canvasNormalEnd.x, canvasNormalEnd.y);
        ctx.strokeStyle = "magenta";
        ctx.lineWidth = normalThickness;
        ctx.stroke();
    });

    // draw target
    const targetRadius = 5;
    const targetPosition = toCanvasPoint(scene.target.position, canvas);
    ctx.beginPath();
    ctx.arc(targetPosition.x, targetPosition.y, targetRadius, 0, 2 * Math.PI);
    ctx.fillStyle = "red";
    ctx.fill();

    // draw rays
    const rays = RayTracer.traceScene(scene);
    rays.forEach(ray => {
        ctx.beginPath();
        ray.forEach(segment => {
            if (segment.type === PathType.Intersection) {
                const start = toCanvasPoint(segment.start, canvas);
                const end = toCanvasPoint(segment.end, canvas);
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
            } else if (segment.type === PathType.Outgoing) {
                const start = toCanvasPoint(segment.start, canvas);
                const end = toCanvasPoint(
                    segment.start.add(segment.direction.scale(OUTGOING_RAY_LENGTH)),
                    canvas,
                );
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
            }
        });
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}