import unittest
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

from inference import (
    apply_global_nms,
    axis_positions,
    decode_image_bytes,
    generate_tiles,
    numpy_nms,
)


class InferenceUtilityTests(unittest.TestCase):
    def test_decode_rgb_tiff(self):
        source = Image.new("RGB", (24, 16), (12, 34, 56))
        buffer = BytesIO()
        source.save(buffer, format="TIFF")

        decoded = decode_image_bytes(buffer.getvalue(), "rgb.tif")

        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (24, 16))
        self.assertEqual(decoded.getpixel((0, 0)), (12, 34, 56))

    def test_decode_16bit_grayscale_tiff(self):
        source_array = np.linspace(0, 65535, 32 * 20, dtype=np.uint16).reshape(20, 32)
        source = Image.fromarray(source_array)
        buffer = BytesIO()
        source.save(buffer, format="TIFF")

        decoded = decode_image_bytes(buffer.getvalue(), "gray16.tiff")
        decoded_array = np.asarray(decoded)

        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (32, 20))
        self.assertEqual(decoded_array.dtype, np.uint8)
        self.assertGreater(int(decoded_array.max()), int(decoded_array.min()))
        self.assertTrue(np.array_equal(decoded_array[..., 0], decoded_array[..., 1]))

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

