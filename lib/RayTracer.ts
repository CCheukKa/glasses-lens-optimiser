import { Vector2 } from "./Vector";

type LightSource = {
    position: Vector2;
    direction: Vector2;
};
type Lens = {
    position: Vector2;
    refractiveIndex: number;
    function: (x: number) => number;
};
type Target = {
    position: Vector2;
};
export class Scene {
    lightSources: LightSource[];
    lens: Lens;
    target: Target;

    constructor(lightSources: LightSource[], lens: Lens, target: Target) {
        this.lightSources = lightSources.map(lightSource => ({
            ...lightSource,
            direction: lightSource.direction.normalised(),
        }));
        this.lens = lens;
        this.target = target;
    }
}