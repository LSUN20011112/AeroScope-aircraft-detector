import unittest

import numpy as np
import pandas as pd
from PIL import Image

from inference import apply_global_nms, axis_positions, generate_tiles, numpy_nms


class InferenceUtilityTests(unittest.TestCase):
    def test_axis_positions_covers_far_edge(self):
        self.assertEqual(axis_positions(3000, 640, 480), [0, 480, 960, 1440, 1920, 2360])

    def test_houston_dimensions_generate_30_tiles(self):
        image = Image.new("RGB", (3000, 2500), "white")
        self.assertEqual(len(generate_tiles(image, 640, 0.25)), 30)

    def test_numpy_nms_suppresses_overlapping_lower_score(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100], [200, 200, 250, 250]], dtype=float)
        scores = np.array([0.95, 0.80, 0.70], dtype=float)
        self.assertEqual(numpy_nms(boxes, scores, 0.40), [0, 2])

    def test_global_nms_preserves_output_columns(self):
        frame = pd.DataFrame(
            [
                {"class_id": 0, "class_name": "aircraft", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 20, "y2": 20, "tile_name": "a"},
                {"class_id": 0, "class_name": "aircraft", "confidence": 0.8, "x1": 1, "y1": 1, "x2": 20, "y2": 20, "tile_name": "b"},
            ]
        )
        result = apply_global_nms(frame, 0.4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["detection_id"], 1)
        self.assertEqual(result.iloc[0]["box_width"], 20)

    def test_empty_global_nms_has_export_schema(self):
        empty = pd.DataFrame(columns=["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "tile_name"])
        result = apply_global_nms(empty, 0.4)
        self.assertEqual(len(result), 0)
        self.assertIn("detection_id", result.columns)
        self.assertIn("box_width", result.columns)


if __name__ == "__main__":
    unittest.main()
