import time

class SpeedEstimator:
    """Frame-to-frame pixel displacement speed estimate, tracked per object
    ID so multiple bikes don't get compared against each other's positions."""

    def __init__(self, scale_factor=0.1):
        self.scale_factor = scale_factor
        self.prev = {}  # track_id -> (center_x, timestamp)

    def calculate_speed(self, track_id, box):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        now = time.time()

        speed = 0

        if track_id in self.prev:
            prev_x, prev_time = self.prev[track_id]
            time_diff = now - prev_time

            if time_diff > 0:
                distance = abs(center_x - prev_x)
                speed = int((distance / time_diff) * self.scale_factor)

        self.prev[track_id] = (center_x, now)

        return speed