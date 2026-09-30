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
type DiscretisedLens = LensSurface[];
type LensSurface = {
    start: Vector2;
    end: Vector2;
    normal: Vector2;
};
export class Scene {
    public lightSources: LightSource[];
    private _lens: Lens;
    public target: Target;
    private _discretisedLens: DiscretisedLens;

    constructor(lightSources: LightSource[], lens: Lens, target: Target) {
        this.lightSources = lightSources.map(lightSource => ({
            ...lightSource,
            direction: lightSource.direction.normalised(),
        }));
        this._lens = lens;
        this.target = target;
        this._discretisedLens = this.discretiseLens();
    }

    get lens() {
        return this._lens;
    }

    set lens(newLens: Lens) {
        this._lens = newLens;
        this._discretisedLens = this.discretiseLens();
    }

    get discretisedLens() {
        return this._discretisedLens;
    }

    private discretiseLens(sampleDensity: number = 10, traceBoundaryRadius: number = 2): DiscretisedLens {
        const points = Array.from({ length: sampleDensity * traceBoundaryRadius }, (_, i) => {
            const x = -traceBoundaryRadius + (i / (sampleDensity * traceBoundaryRadius - 1)) * (2 * traceBoundaryRadius);
            const y = this.lens.function(x);
            return new Vector2(x, y);
        });
        const lensSurfaces: LensSurface[] = [];
        for (let i = 0; i < points.length - 1; i++) {
            const start = points[i];
            const end = points[i + 1];
            const normal = end.subtract(start).normalised().perpendicular();
            lensSurfaces.push({ start, end, normal });
        }
        return lensSurfaces;
    }
}

type Path = {
    start: Vector2;
    end: Vector2;
} | {
    start: Vector2;
    direction: Vector2;
};
type Ray = Path[];
export class RayTracer {
    public traceBoundaryX = 2;
    public traceBoundaryY = 2;

    public traceScene(scene: Scene): Ray[] {
        return scene.lightSources.map(lightSource => this.traceRay(lightSource, scene.discretisedLens));
    }
    private traceRay(lightSource: LightSource, lens: DiscretisedLens): Ray {
        return [];
    }
}