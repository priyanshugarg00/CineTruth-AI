import os
import cv2
from PIL import Image
from PIL.ExifTags import TAGS


class MetadataAgent:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def extract_metadata(self, media_path: str) -> dict:
        ext = os.path.splitext(str(media_path))[1].lower()
        if ext in self.VIDEO_EXTENSIONS:
            return self._extract_video_metadata(media_path)
        return self._extract_image_metadata(media_path)

    def _extract_image_metadata(self, image_path: str) -> dict:
        metadata = {"media_type": "image"}
        try:
            with Image.open(image_path) as image:
                exif_data = image.getexif()
                if exif_data:
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        metadata[str(tag_name)] = str(value)

                metadata["dimensions"] = f"{image.width}x{image.height}"
                metadata["width"] = image.width
                metadata["height"] = image.height
                metadata["format"] = image.format
                metadata["mode"] = image.mode
        except Exception as exc:
            metadata["error"] = f"Failed to extract image metadata: {exc}"
        return metadata

    def _extract_video_metadata(self, video_path: str) -> dict:
        metadata = {"media_type": "video"}
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                metadata["error"] = "Could not open video file."
                return metadata

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = (frame_count / fps) if fps > 0 else 0

            metadata.update(
                {
                    "dimensions": f"{width}x{height}",
                    "width": width,
                    "height": height,
                    "fps": round(fps, 3),
                    "frame_count": frame_count,
                    "duration_seconds": round(duration, 3),
                    "file_size_bytes": os.path.getsize(video_path)
                    if os.path.exists(video_path)
                    else 0,
                }
            )
        except Exception as exc:
            metadata["error"] = f"Failed to extract video metadata: {exc}"
        finally:
            cap.release()
        return metadata
