import { describe, it, expect } from "vitest";
import { checkArbitrage } from "./arbitrage";

describe("checkArbitrage", () => {
  const TGrid = [0.1, 0.5, 1.0];
  const KGrid = [-0.2, 0.0, 0.2];

  it("should report no arbitrage for a flat and consistent implied volatility surface", () => {
    // 0.20 flat volatility (no skew, no smile, no maturity term decay)
    const ivSurface = [
      [0.2, 0.2, 0.2],
      [0.2, 0.2, 0.2],
      [0.2, 0.2, 0.2],
    ];

    const result = checkArbitrage(ivSurface, TGrid, KGrid);
    expect(result.hasCalendarArb).toBe(false);
    expect(result.hasButterflyArb).toBe(false);
    expect(result.calendarViolations.flat().some(Boolean)).toBe(false);
    expect(result.butterflyViolations.flat().some(Boolean)).toBe(false);
  });

  it("should detect calendar spread arbitrage when variance decreases over maturity", () => {
    // Variance at T=0.1 (K=0.0) is 0.40^2 * 0.1 = 0.016
    // Variance at T=0.5 (K=0.0) is 0.15^2 * 0.5 = 0.01125 (decreased!)
    const ivSurface = [
      [0.4, 0.4, 0.4],
      [0.15, 0.15, 0.15],
      [0.2, 0.2, 0.2],
    ];

    const result = checkArbitrage(ivSurface, TGrid, KGrid);
    expect(result.hasCalendarArb).toBe(true);
    expect(result.calendarViolations[1][0]).toBe(true); // T=0.5 has violation
  });

  it("should detect butterfly spread arbitrage when density is negative (arbitrary smile violation)", () => {
    // Create an extreme frown with high vol at center, low vol at wings
    const ivSurface = [
      [0.2, 2.0, 0.2],
      [0.2, 2.0, 0.2],
      [0.2, 2.0, 0.2],
    ];

    const result = checkArbitrage(ivSurface, TGrid, KGrid);
    expect(result.hasButterflyArb).toBe(true);
    expect(result.butterflyViolations[0][1]).toBe(true); // center strike violated
  });
});
