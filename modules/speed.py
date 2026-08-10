import time


def calibrate_pixels_per_meter(pixel_distance, real_world_meters):
    """Compute the pixels_per_meter calibration value SpeedEstimator needs.

    Mark two points in your camera's frame that are a known real-world
    distance apart (e.g. two road markings), measure the pixel distance
    between them in a frame, and pass both here:

        calibrate_pixels_per_meter(pixel_distance=340, real_world_meters=5)
    """
    if real_world_meters <= 0:
        raise ValueError("real_world_meters must be > 0")
    return pixel_distance / real_world_meters


class SpeedEstimator:
    """Frame-to-frame pixel displacement speed estimate, tracked per object
    ID so multiple bikes don't get compared against each other's positions.

    Requires camera calibration (`pixels_per_meter`) to produce a real
    speed in km/h -- without it, pixel displacement has no physical unit.
    Get this value from calibrate_pixels_per_meter() using a known
    real-world reference distance in your camera's frame. This is also
    only valid for motion roughly perpendicular to the camera; a vehicle
    moving toward/away from the camera will read artificially slow.
    """

    def __init__(self, pixels_per_meter):
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be > 0")

        self.pixels_per_meter = pixels_per_meter
        self.prev = {}  # track_id -> (center_x, timestamp)

    def calculate_speed(self, track_id, box):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        now = time.time()

        speed_kmh = 0

        if track_id in self.prev:
            prev_x, prev_time = self.prev[track_id]
            time_diff = now - prev_time

            if time_diff > 0:
                pixel_distance = abs(center_x - prev_x)
                meters = pixel_distance / self.pixels_per_meter
                speed_kmh = int((meters / time_diff) * 3.6)  # m/s -> km/h

        self.prev[track_id] = (center_x, now)

        return speed_kmh
