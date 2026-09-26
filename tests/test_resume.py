from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / 'distill'))
from catalog import Species
from download_inaturalist import download_species, write_manifest
from teacher import MegaDetector


class ResumeTests(unittest.TestCase):
    def test_existing_photo_recovers_attribution_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / '000-1'
            folder.mkdir()
            (folder / '42-7.jpg').write_bytes(b'existing image')
            payload = {'results': [{'id': 42, 'photos': [
                {'id': 7, 'url': 'https://example.org/square.jpg',
                 'license_code': 'cc-by', 'attribution': 'Photographer'}]}]}
            with patch('download_inaturalist.api_get', return_value=payload), patch('download_inaturalist.download') as download:
                rows = download_species({'inaturalist_place_id': 30}, Species(1, 'Test species', 'Test Animal', 'Aves', 20), 0, root, 20)
                download.assert_not_called()
            write_manifest(root, rows)
            with (root / 'attribution.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]['attribution'], 'Photographer')
            self.assertEqual(saved[0]['license'], 'cc-by')
            self.assertFalse(list(root.glob('*.tmp')))

    def test_detector_converts_pil_rgb_to_ultralytics_bgr(self):
        detector = MegaDetector.__new__(MegaDetector)
        detector.device = 'cpu'
        detector.model = Mock()
        detector.model.predict.return_value = []
        detector.detect_many([Image.new('RGB', (2, 2), (255, 10, 30))], .25)
        arrays = detector.model.predict.call_args.args[0]
        np.testing.assert_array_equal(arrays[0][0, 0], [30, 10, 255])


if __name__ == '__main__':
    unittest.main()
