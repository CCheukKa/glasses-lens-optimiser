import { Vector2 } from "./MathExtra";

type LightSource = {
    position: Vector2;
    direction: Vector2;
};
type Lens = {
    position: Vector2;
    refractiveIndex: number;
    readonly function: (x: number) => number[];
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

    public setLensFunction(newFunction: (x: number) => number[]) {
        this._lens = { ...this._lens, function: newFunction };
        this._discretisedLens = this.discretiseLens();
    }

    private discretiseLens(sampleDensity: number = 100, traceBoundaryRadius: number = 2, surfaceThickness: number = 0.5): DiscretisedLens {
        const pointCount = Math.ceil(Math.max(2, sampleDensity * traceBoundaryRadius));

        // Prediction output as y values for the lens function at evenly spaced x values
        // const points = Array.from({ length: pointCount }, (_, i) => {
        //     const x = -traceBoundaryRadius + (i / (pointCount - 1)) * (2 * traceBoundaryRadius);
        //     const y = this.lens.function(x);
        //     return this.lens.position.add(new Vector2(x, y));
        // });

        // Prediction output as slopes relative to the fixed midpoint of the lens.
        const topSurfacePoints = Array.from({ length: pointCount }, (_, i) => {
            const relativeX = -traceBoundaryRadius + (i / (pointCount - 1)) * (2 * traceBoundaryRadius);
            const slope = this.lens.function(relativeX)[0];
            const relativeY = slope * relativeX;
            return this.lens.position.add(new Vector2(relativeX, relativeY + surfaceThickness / 2));
        });
        const bottomSurfacePoints = Array.from({ length: pointCount }, (_, i) => {
            const relativeX = -traceBoundaryRadius + (i / (pointCount - 1)) * (2 * traceBoundaryRadius);
            const slope = this.lens.function(relativeX)[1];
            const relativeY = slope * relativeX;
            return this.lens.position.add(new Vector2(relativeX, relativeY - surfaceThickness / 2));
        });

        const lensSurfaces: LensSurface[] = [];
        for (let i = 0; i < topSurfacePoints.length - 1; i++) {
            const start = topSurfacePoints[i];
            const end = topSurfacePoints[i + 1];
            const normal = end.subtract(start).normalised().perpendicular();
            lensSurfaces.push({ start, end, normal });
        }
        for (let i = 0; i < bottomSurfacePoints.length - 1; i++) {
            const start = bottomSurfacePoints[i];
            const end = bottomSurfacePoints[i + 1];
            const normal = end.subtract(start).normalised().perpendicular().scale(-1);
            lensSurfaces.push({ start, end, normal });
        }
        return lensSurfaces;
    }
}

export enum PathType {
    Intersection = "Intersection",
    Outgoing = "Outgoing",
}
type IntersectionPath = { type: PathType.Intersection; start: Vector2; end: Vector2; };
type OutgoingPath = { type: PathType.Outgoing; start: Vector2; direction: Vector2 };
type Ray = [...IntersectionPath[], OutgoingPath];
export class RayTracer {
    public static traceScene(scene: Scene): Ray[] {
        return scene.lightSources.map(lightSource =>
            this.traceRay(lightSource, scene.discretisedLens, scene.lens.refractiveIndex),
        );
    }

    private static traceRay(lightSource: LightSource, lens: DiscretisedLens, refractiveIndex: number): Ray {
        const paths: IntersectionPath[] = [];
        let start = lightSource.position;
        let direction = lightSource.direction;

        for (let interaction = 0; interaction < 2; interaction++) {
            const nearestIntersection = this.findNearestIntersection(start, direction, lens);
            if (!nearestIntersection) {
                break;
            }

            paths.push({
                type: PathType.Intersection,
                start,
                end: nearestIntersection.point,
            });

            const normal = nearestIntersection.surface.normal;
            const enteringGlass = direction.dot(normal) < 0;
            const eta = enteringGlass ? 1 / refractiveIndex : refractiveIndex;
            direction = this.refract(direction, normal, eta);
            start = nearestIntersection.point;
        }

        return [
            ...paths,
            { type: PathType.Outgoing, start, direction },
        ];
    }

    private static findNearestIntersection(
        origin: Vector2,
        direction: Vector2,
        lens: DiscretisedLens,
    ): { point: Vector2; surface: LensSurface; distance: number } | undefined {
        let nearestIntersection: { point: Vector2; surface: LensSurface; distance: number } | undefined;
        for (const surface of lens) {
            const intersection = this.intersectRayWithSurface(origin, direction, surface);
            if (intersection && (!nearestIntersection || intersection.distance < nearestIntersection.distance)) {
                nearestIntersection = { ...intersection, surface };
            }
        }
        return nearestIntersection;
    }

    private static refract(direction: Vector2, surfaceNormal: Vector2, eta: number): Vector2 {
        const normal = direction.dot(surfaceNormal) > 0
            ? surfaceNormal.scale(-1)
            : surfaceNormal;
        const cosineIncident = -direction.dot(normal);
        const sineSquaredTransmitted = eta * eta * (1 - cosineIncident * cosineIncident);
        if (sineSquaredTransmitted > 1) {
            return direction.subtract(normal.scale(2 * direction.dot(normal))).normalised();
        }

        return direction
            .scale(eta)
            .add(normal.scale(eta * cosineIncident - Math.sqrt(1 - sineSquaredTransmitted)))
            .normalised();
    }

    private static intersectRayWithSurface(
        origin: Vector2,
        direction: Vector2,
        surface: LensSurface,
    ): { point: Vector2; distance: number } | undefined {
        const surfaceDirection = surface.end.subtract(surface.start);
        const denominator = direction.x * surfaceDirection.y - direction.y * surfaceDirection.x;
        if (Math.abs(denominator) < 1e-10) {
            return undefined;
        }

        const offset = surface.start.subtract(origin);
        const rayDistance = (offset.x * surfaceDirection.y - offset.y * surfaceDirection.x) / denominator;
        const surfaceDistance = (offset.x * direction.y - offset.y * direction.x) / denominator;
        if (rayDistance < 1e-10 || surfaceDistance < -1e-10 || surfaceDistance > 1 + 1e-10) {
            return undefined;
        }

        return {
            point: origin.add(direction.scale(rayDistance)),
            distance: rayDistance,
        };
    }
}