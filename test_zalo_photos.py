"""Photo delivery checks, without uploading or sending external messages."""

import json
import unittest
from email.message import Message
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import camera_fire_temporal_zalo as camera


class PhotoDeliveryTests(unittest.TestCase):
    def response(self, mime, body):
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = Message()
        response.headers["Content-Type"] = mime
        response.read.return_value = body
        return response

    def test_valid_jpeg_is_accepted(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((32, 48, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with patch.object(camera.urllib.request, "urlopen",
                          return_value=self.response("image/jpeg", encoded.tobytes())):
            camera.check_photo_url("https://i.ibb.co/example.jpg")

    def test_html_error_page_is_rejected_even_with_http_success(self):
        with patch.object(camera.urllib.request, "urlopen",
                          return_value=self.response("text/html", b"<html>Expired</html>")):
            with self.assertRaises(RuntimeError):
                camera.check_photo_url("https://i.ibb.co/example.jpg")

    def test_corrupt_jpeg_is_rejected(self):
        with patch.object(camera.urllib.request, "urlopen",
                          return_value=self.response("image/jpeg", b"not a JPEG")):
            with self.assertRaises(RuntimeError):
                camera.check_photo_url("https://i.ibb.co/example.jpg")

    def test_upload_uses_original_image_url_and_configured_expiration(self):
        result = {"success": True, "data": {
            "url": "https://i.ibb.co/other.jpg",
            "url_viewer": "https://ibb.co/viewer",
            "image": {"url": "https://i.ibb.co/original.jpg"},
        }}
        with patch.object(camera, "read_json_response", return_value=result) as read:
            url = camera.upload_photo({"imgbb_api_key": "fake-key"}, b"jpeg")
        self.assertEqual(url, "https://i.ibb.co/original.jpg")
        fields = camera.urllib.parse.parse_qs(read.call_args.args[0].data.decode("ascii"))
        self.assertEqual(fields["expiration"], [str(camera.PHOTO_EXPIRATION_SECONDS)])

    def test_photo_request_includes_direct_link_without_losing_it_to_truncation(self):
        url = "https://i.ibb.co/original.jpg"
        with patch.object(camera, "read_json_response", return_value={
            "ok": True, "result": {"message_id": "test-id"},
        }) as read:
            message_id = camera.send_zalo_photo(
                {"chat_id": "fake-chat", "bot_token": "fake-token"}, url, "a" * 2100,
            )
        payload = json.loads(read.call_args.args[0].data)
        self.assertEqual(payload["photo"], url)
        self.assertIn(url, payload["caption"])
        self.assertLessEqual(len(payload["caption"]), 2000)
        self.assertEqual(message_id, "test-id")

    def test_zalo_rejection_is_not_reported_as_success(self):
        with patch.object(camera, "read_json_response", return_value={"ok": False}):
            with self.assertRaises(RuntimeError):
                camera.send_zalo_photo(
                    {"chat_id": "fake-chat", "bot_token": "fake-token"},
                    "https://i.ibb.co/original.jpg", "test",
                )


if __name__ == "__main__":
    unittest.main()
