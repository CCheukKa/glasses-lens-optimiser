export class Vector2 {
    public x: number;
    public y: number;

    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    magnitude(): number {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }
    normalised(): Vector2 {
        const mag = this.magnitude();
        if (mag === 0) {
            return new Vector2(0, 0);
        }
        return new Vector2(this.x / mag, this.y / mag);
    }

    add(other: Vector2): Vector2 {
        return new Vector2(this.x + other.x, this.y + other.y);
    }
    dot(other: Vector2): number {
        return this.x * other.x + this.y * other.y;
    }
}