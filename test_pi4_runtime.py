"""Pi runtime integration checks without a camera, upload, or message."""

from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import camera_pi4 as runner


class PiRuntimeTests(unittest.TestCase):
    def test_rectangular_input_reaches_backend_with_thread_setting(self):
        with TemporaryDirectory(suffix="_ncnn_model") as directory:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            camera = MagicMock()
            camera.isOpened.return_value = True
            camera.read.return_value = (True, image)
            cv2 = MagicMock()
            cv2.VideoCapture.return_value = camera
            result = SimpleNamespace(boxes=None, speed={
                "preprocess": 2.0, "inference": 120.0, "postprocess": 1.0,
            })
            model = MagicMock()
            ncnn = SimpleNamespace(Net=lambda: SimpleNamespace(opt=SimpleNamespace(num_threads=4)))
            original_net = ncnn.Net
            net = None

            def predict(**kwargs):
                nonlocal net
                if net is None:
                    net = ncnn.Net()
                    # Loading must see the requested threads, not a later change.
                    self.assertEqual(net.opt.num_threads, 2)
                callback = model.add_callback.call_args.args[1]
                callback(SimpleNamespace(imgsz=kwargs["imgsz"], model=SimpleNamespace(net=net)))
                self.assertEqual(net.opt.num_threads, 2)
                return [result]

            model.predict.side_effect = predict
            ultralytics = SimpleNamespace(YOLO=MagicMock(return_value=model))
            torch = MagicMock()
            output = StringIO()
            with patch.dict("sys.modules", {"cv2": cv2, "torch": torch, "ultralytics": ultralytics, "ncnn": ncnn}), \
                    patch.object(runner.sys, "platform", "linux"), \
                    patch.object(runner.signal, "signal"), \
                    patch.object(runner.alerts, "WINDOW_SECONDS", runner.alerts.WINDOW_SECONDS), \
                    patch.object(runner.alerts, "MAX_FRAME_GAP", runner.alerts.MAX_FRAME_GAP), \
                    patch.object(runner.alerts, "load_zalo_config") as load_config, \
                    redirect_stdout(output):
                runner.main(["--model", directory, "--imgsz", "320", "416",
                             "--ncnn-threads", "2", "--no-zalo", "--max-frames", "2"])

            self.assertEqual(model.predict.call_count, 2)
            self.assertEqual(model.predict.call_args.kwargs["imgsz"], [320, 416])
            self.assertFalse(model.predict.call_args.kwargs["rect"])
            load_config.assert_not_called()
            cv2.imshow.assert_not_called()
            camera.release.assert_called_once()
            self.assertIs(ncnn.Net, original_net)
            self.assertIn("NCNN: input=320x416; threads=2", output.getvalue())
            self.assertIn("infer_ms=120.0", output.getvalue())

    def test_export_shape_mismatch_fails_before_inference(self):
        model = MagicMock()
        args = SimpleNamespace(imgsz=[320, 416], ncnn_threads=4)
        runner.configure_ncnn(model, args)
        callback = model.add_callback.call_args.args[1]
        net = SimpleNamespace(opt=SimpleNamespace(num_threads=2))
        with self.assertRaisesRegex(ValueError, "NCNN input"):
            callback(SimpleNamespace(imgsz=[416, 416], model=SimpleNamespace(net=net)))
        self.assertEqual(net.opt.num_threads, 2)

    def test_rectangular_letterbox_preserves_scene_scale(self):
        from ultralytics.data.augment import LetterBox

        # Non-padding pixels must be exactly the same, at the same scale.
        image = np.full((480, 640, 3), 200, dtype=np.uint8)
        square = LetterBox((416, 416), auto=False)(image=image)
        rectangle = LetterBox((320, 416), auto=False)(image=image)
        np.testing.assert_array_equal(square[52:364], rectangle[4:316])
        self.assertEqual(rectangle.shape, (320, 416, 3))

    def test_existing_square_cli_remains_supported(self):
        args = runner.parse_args(["--imgsz", "416"])
        self.assertEqual(args.imgsz, [416, 416])

    def test_ncnn_constructor_restored_on_loading_failure(self):
        ncnn = SimpleNamespace(Net=lambda: SimpleNamespace(opt=SimpleNamespace(num_threads=4)))
        original_net = ncnn.Net
        with patch.dict("sys.modules", {"ncnn": ncnn}):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                with runner.ncnn_thread_options(2):
                    self.assertEqual(ncnn.Net().opt.num_threads, 2)
                    raise RuntimeError("load failed")
        self.assertIs(ncnn.Net, original_net)


if __name__ == "__main__":
    unittest.main()
