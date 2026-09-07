import os
import cv2


class FrameAgent:
    def __init__(self, seconds_interval=2, max_frames=8):
        self.seconds_interval = max(1, int(seconds_interval))
        self.max_frames = max(1, int(max_frames))

    def extract_frames(self, video_path: str, output_dir="temp_frames") -> list:
        """Extract a small representative frame sample from a video."""
        os.makedirs(output_dir, exist_ok=True)
        extracted_files = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return extracted_files

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_rate = max(1, int(fps * self.seconds_interval))

        count = 0
        saved_count = 0
        while cap.isOpened() and saved_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_rate == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_count}.jpg")
                if cv2.imwrite(frame_filename, frame):
                    extracted_files.append(frame_filename)
                    saved_count += 1
            count += 1

        cap.release()
        print(f"[FrameAgent] Extracted {len(extracted_files)} frame(s) from video.")
        return extracted_files
