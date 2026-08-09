import math


class CentroidTracker:
    """Assigns a persistent ID to each box across frames by matching to the
    nearest previous centroid (greedy, smallest distance first). Objects
    unmatched for `max_disappeared` frames are dropped."""

    def __init__(self, max_disappeared=10, max_distance=80):
        self.next_id = 0
        self.objects = {}       # id -> centroid (cx, cy)
        self.boxes = {}         # id -> box (x1, y1, x2, y2)
        self.disappeared = {}   # id -> frames since last matched
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, boxes):
        """Returns {id: box} for objects matched in *this* frame only.
        Unmatched previous objects are kept internally (so a briefly
        occluded object keeps its ID if it reappears) but are not
        returned until they're matched again."""

        if not boxes:
            for object_id in list(self.disappeared):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return {}

        input_centroids = [self._centroid(box) for box in boxes]
        active_ids = set()

        if not self.objects:
            for centroid, box in zip(input_centroids, boxes):
                active_ids.add(self._register(centroid, box))
            return {oid: self.boxes[oid] for oid in active_ids}

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        pairs = []
        for row, object_centroid in enumerate(object_centroids):
            for col, input_centroid in enumerate(input_centroids):
                distance = self._distance(object_centroid, input_centroid)
                pairs.append((distance, row, col))

        pairs.sort(key=lambda p: p[0])

        used_rows = set()
        used_cols = set()

        for distance, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue
            if distance > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.boxes[object_id] = boxes[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)
            active_ids.add(object_id)

        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            active_ids.add(self._register(input_centroids[col], boxes[col]))

        return {oid: self.boxes[oid] for oid in active_ids}

    def _register(self, centroid, box):
        object_id = self.next_id
        self.objects[object_id] = centroid
        self.boxes[object_id] = box
        self.disappeared[object_id] = 0
        self.next_id += 1
        return object_id

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]

    @staticmethod
    def _centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
