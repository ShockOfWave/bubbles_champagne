import cv2
import numpy as np
from ultralytics import YOLO
import os
from src.utils import segmentation_model_path
from tqdm import tqdm

class VideoSegmenter:
    def __init__(self):
        """
        Initialize the VideoSegmenter with YOLO model
        """
        self._model = YOLO(segmentation_model_path)
        self._conf = 0.3

    def process_frame(self, frame):
        """
        Process a single frame and generate mask
        Args:
            frame (np.ndarray): Input frame
        Returns:
            np.ndarray: Frame with drawn polygons
        """
        # Get predictions from YOLO model
        results = self._model.predict(frame, conf=self._conf, verbose=False)
        
        # Create a copy of the frame to draw on
        result_frame = frame.copy()
        
        # Process each detection
        for result in results:
            # Get segments (polygons)
            if result.masks is not None:
                for segment in result.masks.xy:
                    # Convert segment points to numpy array of integers
                    points = np.array(segment, dtype=np.int32)
                    
                    # Reshape points for cv2.polylines
                    points = points.reshape((-1, 1, 2))
                    
                    # Draw the polygon in light blue color
                    cv2.polylines(result_frame, [points], True, (255, 255, 0), 2)
        
        return result_frame

    def process_video(self, input_video_path, output_video_path):
        """
        Process entire video and save result
        Args:
            input_video_path (str): Path to input video
            output_video_path (str): Path to save processed video
        """
        # Open video file
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        try:
            # Create progress bar for frames
            with tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_video_path)}") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Write frame
                    out.write(processed_frame)
                    
                    # Update progress bar
                    pbar.update(1)
                
        finally:
            cap.release()
            out.release()
