import time

class SpeedEstimator:
    def __init__(self):
        self.prev_time = time.time()
        self.prev_positions = []

    def calculate_speed(self, bikes):
        current_time = time.time()

        if not bikes:
            return 0

        speed = 0

        for bike in bikes:
            x1, y1, x2, y2 = bike
            center = (x1 + x2) // 2

            if self.prev_positions:
                distance = abs(center - self.prev_positions[0])
                time_diff = current_time - self.prev_time

                if time_diff > 0:
                    speed = int((distance / time_diff) * 0.1)  # scale factor

        self.prev_positions = [center]
        self.prev_time = current_time

        return speed