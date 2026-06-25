/**
 * Static arbitrage checking utilities for implied volatility surfaces.
 * Implements calendar spread and butterfly (Durrleman) arbitrage checks.
 */

export interface ArbitrageResult {
  hasCalendarArb: boolean;
  hasButterflyArb: boolean;
  calendarViolations: boolean[][]; // Shape: (nT, nK)
  butterflyViolations: boolean[][]; // Shape: (nT, nK)
  gValues: number[][]; // Shape: (nT, nK)
}

/**
 * Checks an implied volatility surface for calendar and butterfly arbitrage violations.
 * @param ivSurface 2D array of shape (nT, nK) with IV values.
 * @param TGrid Array of maturities of length nT.
 * @param KGrid Array of log-strikes of length nK.
 */
export function checkArbitrage(
  ivSurface: number[][],
  TGrid: number[],
  KGrid: number[]
): ArbitrageResult {
  const nT = ivSurface.length;
  const nK = ivSurface[0]?.length || 0;

  const calendarViolations = Array.from({ length: nT }, () => Array(nK).fill(false));
  const butterflyViolations = Array.from({ length: nT }, () => Array(nK).fill(false));
  const gValues = Array.from({ length: nT }, () => Array(nK).fill(1));

  let hasCalendarArb = false;
  let hasButterflyArb = false;

  // 1. Calculate total variance: w(T, k) = iv^2 * T
  const w: number[][] = [];
  for (let i = 0; i < nT; i++) {
    w.push([]);
    for (let j = 0; j < nK; j++) {
      w[i].push(Math.pow(ivSurface[i][j], 2) * TGrid[i]);
    }
  }

  // 2. Calendar Arbitrage Check
  // w(T_i+1, k) - w(T_i, k) must be >= 0 (with a small tolerance)
  for (let i = 0; i < nT - 1; i++) {
    for (let j = 0; j < nK; j++) {
      if (w[i + 1][j] - w[i][j] < -1e-8) {
        calendarViolations[i + 1][j] = true;
        hasCalendarArb = true;
      }
    }
  }

  // 3. Butterfly Arbitrage Check (Durrleman's condition)
  // Calculate derivatives w' and w'' w.r.t log-strike k
  const dk: number[] = [];
  for (let j = 0; j < nK - 1; j++) {
    dk.push(KGrid[j + 1] - KGrid[j]);
  }

  for (let i = 0; i < nT; i++) {
    const wPrime = Array(nK).fill(0);
    const wPrimePrime = Array(nK).fill(0);

    // Central differences for interior points (1 to nK - 2)
    for (let j = 1; j < nK - 1; j++) {
      const h_l = dk[j - 1];
      const h_r = dk[j];
      wPrime[j] = (w[i][j + 1] - w[i][j - 1]) / (h_l + h_r);
      wPrimePrime[j] = 2.0 * ((w[i][j + 1] - w[i][j]) / h_r - (w[i][j] - w[i][j - 1]) / h_l) / (h_l + h_r);
    }

    // Boundary points (first/last strike)
    if (nK > 1) {
      wPrime[0] = (w[i][1] - w[i][0]) / dk[0];
      wPrime[nK - 1] = (w[i][nK - 1] - w[i][nK - 2]) / dk[dk.length - 1];
      wPrimePrime[0] = wPrimePrime[1];
      wPrimePrime[nK - 1] = wPrimePrime[nK - 2];
    }

    // Compute Durrleman's g(k)
    for (let j = 0; j < nK; j++) {
      const wVal = Math.max(w[i][j], 1e-9); // clamp to avoid division by zero
      const term1 = Math.pow(1.0 - (KGrid[j] * wPrime[j]) / (2.0 * wVal), 2);
      const term2 = (Math.pow(wPrime[j], 2) / 4.0) * (1.0 / wVal + 0.25);
      const term3 = wPrimePrime[j] / 2.0;

      const g = term1 - term2 + term3;
      gValues[i][j] = g;

      // Durrleman violation check (ignore boundary coordinates for second derivative)
      if (j > 0 && j < nK - 1) {
        if (g < -1e-8) {
          butterflyViolations[i][j] = true;
          hasButterflyArb = true;
        }
      }
    }
  }

  return {
    hasCalendarArb,
    hasButterflyArb,
    calendarViolations,
    butterflyViolations,
    gValues,
  };
}
