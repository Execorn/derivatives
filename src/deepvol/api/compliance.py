import os
import time
import math
import logging
import threading
from typing import List, Tuple

logger = logging.getLogger("deepvol.api.compliance")

class ComplianceMonitor:
    """
    Automated Compliance Monitor for SR 26-2.
    Tracks out-of-distribution (OOD) queries, clamps parameters to boundaries,
    and computes the Population Stability Index (PSI) to detect input drift.
    """
    def __init__(self, log_path: str = "logs/compliance_audit.log"):
        self.log_path = log_path
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.batch_T: List[float] = []
        self.batch_k: List[float] = []
        self.batch_size = 1000

        # Define bin edges for T (8 bins total)
        self.T_edges = [0.2, 0.45, 0.75, 1.05, 1.35, 1.65, 1.9]
        self.T_expected = [0.125] * 8

        # Define bin edges for k (11 bins total)
        self.k_edges = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        self.k_expected = [1.0 / 11] * 11

    def audit_ood_and_clamp(self, T: float, K: float, S0: float, model_name: str) -> Tuple[float, float, bool]:
        """
        Check if (T, K) is OOD (outside convex hull: T in [0.1, 2.0], log-moneyness in [-0.5, 0.5]).
        Clamps inputs if OOD and writes to audit log.
        Returns: (T_clamped, K_clamped, is_ood)
        """
        k = math.log(K / S0) if S0 > 0 else 0.0
        is_ood = False
        T_clamped = T
        k_clamped = k

        if T < 0.1:
            T_clamped = 0.1
            is_ood = True
        elif T > 2.0:
            T_clamped = 2.0
            is_ood = True

        if k < -0.5:
            k_clamped = -0.5
            is_ood = True
        elif k > 0.5:
            k_clamped = 0.5
            is_ood = True

        K_clamped = S0 * math.exp(k_clamped)

        if is_ood:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            log_msg = (
                f"[{timestamp}] OOD WARNING: Input (T={T:.4f}, K={K:.4f}, S0={S0:.2f}) "
                f"is outside training convex hull. Clamping to (T={T_clamped:.4f}, K={K_clamped:.4f}). "
                f"Model: {model_name}\n"
            )
            logger.warning(log_msg.strip())
            with self.lock:
                try:
                    with open(self.log_path, "a") as f:
                        f.write(log_msg)
                except Exception as e:
                    logger.error(f"Failed to write to compliance audit log: {e}")

        # Track query for drift analysis
        self.track_query(T_clamped, k_clamped)

        return T_clamped, K_clamped, is_ood

    def track_query(self, T: float, k: float):
        """Track query (T, k) for PSI computation."""
        with self.lock:
            self.batch_T.append(T)
            self.batch_k.append(k)
            if len(self.batch_T) >= self.batch_size:
                T_batch = self.batch_T[:self.batch_size]
                k_batch = self.batch_k[:self.batch_size]
                self.batch_T = self.batch_T[self.batch_size:]
                self.batch_k = self.batch_k[self.batch_size:]
                
                # Compute PSI in a background thread to prevent latency spikes
                threading.Thread(target=self._compute_and_log_psi, args=(T_batch, k_batch), daemon=True).start()

    def _compute_and_log_psi(self, T_batch: List[float], k_batch: List[float]):
        # T PSI
        T_counts = [0] * 8
        for t in T_batch:
            bin_idx = 0
            for edge in self.T_edges:
                if t >= edge:
                    bin_idx += 1
                else:
                    break
            T_counts[bin_idx] += 1
        
        # k PSI
        k_counts = [0] * 11
        for k in k_batch:
            bin_idx = 0
            for edge in self.k_edges:
                if k >= edge:
                    bin_idx += 1
                else:
                    break
            k_counts[bin_idx] += 1

        psi_T = self._calculate_psi(T_counts, self.T_expected)
        psi_k = self._calculate_psi(k_counts, self.k_expected)

        if psi_T > 0.2 or psi_k > 0.2:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            warn_msg = (
                f"[{timestamp}] COMPLIANCE WARNING: Online drift detected (SR 26-2)! "
                f"PSI_T={psi_T:.4f}, PSI_k={psi_k:.4f} (threshold=0.2)\n"
            )
            logger.warning(warn_msg.strip())
            with self.lock:
                try:
                    with open(self.log_path, "a") as f:
                        f.write(warn_msg)
                except Exception:
                    pass

    def _calculate_psi(self, counts: List[int], expected_probs: List[float]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        
        # Smooth actual probabilities to prevent log(0) or division by zero
        eps = 1e-5
        actual_probs = [(c + eps) / (total + len(counts) * eps) for c in counts]
        
        psi = 0.0
        for act, exp in zip(actual_probs, expected_probs):
            psi += (act - exp) * math.log(act / exp)
        return psi

# Global compliance monitor instance
compliance_monitor = ComplianceMonitor()
