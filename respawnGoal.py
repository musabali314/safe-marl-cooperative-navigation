import random
import numpy as np

# World boundaries (must match env_coop.py)
X_MIN, X_MAX = -3.0, 3.0
Y_MIN, Y_MAX = -3.0, 3.0

# Safety thresholds
GOAL_MIN_DIST_FROM_OBS = 0.6     # min clearance from obstacles
GOAL_MIN_DIST_FROM_ROBOT = 0.6    # min clearance from robots
MAX_SAMPLING_ATTEMPTS = 300        # prevents infinite loops


class RespawnGoal:
    def __init__(self):
        """
        Equivalent structure to original Respawn() class.
        """
        self.exists = False
        self.goal_xy = np.zeros(2, dtype=np.float32)

    # ----------------------------------------------------------
    def deleteModel(self):
        """Marks goal as deleted (no-op in Python version)."""
        self.exists = False

    # ----------------------------------------------------------
    def respawnModel(self):
        """Marks goal as present."""
        self.exists = True

    # ----------------------------------------------------------
    def _sample_safe_goal(self, robot_positions, obstacles):
        for _ in range(MAX_SAMPLING_ATTEMPTS):

            gx = random.uniform(X_MIN + 0.8, X_MAX - 0.8)
            gy = random.uniform(Y_MIN + 0.8, Y_MAX - 0.8)
            gxy = np.array([gx, gy], dtype=np.float32)

            valid = True

            # ---- Avoid obstacles ----
            for (ox, oy, r) in obstacles:
                dist = np.linalg.norm(gxy - np.array([ox, oy]))
                if dist < (r + GOAL_MIN_DIST_FROM_OBS):
                    valid = False
                    break
            if not valid:
                continue

            # ---- Avoid robots ----
            for pos in robot_positions:
                if np.linalg.norm(gxy - pos) < GOAL_MIN_DIST_FROM_ROBOT:
                    valid = False
                    break
            if not valid:
                continue

            return gxy

        # ----------------------------------------------------------
        # If all attempts fail, fallback to world center
        # ----------------------------------------------------------
        return np.array([0.0, 0.0], dtype=np.float32)

    # ----------------------------------------------------------
    def getPosition(self, robot_positions, obstacles, delete=False):
        """
        Args:
            robot_positions: np.array(N, 2)
            obstacles: [(ox, oy, r), ...]
            delete: whether to delete the previous goal first

        Returns:
            (goal_x, goal_y)
        """

        if delete:
            self.deleteModel()
        else:
            if self.exists:
                self.deleteModel()

        # Sample and respawn new goal
        self.goal_xy = self._sample_safe_goal(robot_positions, obstacles)
        self.respawnModel()

        return float(self.goal_xy[0]), float(self.goal_xy[1])
