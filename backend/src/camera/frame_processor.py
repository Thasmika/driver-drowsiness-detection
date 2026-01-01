"""
Real-Time Frame Processor

This module provides continuous video frame processing for drowsiness detection,
integrating face detection, feature extraction, ML inference, and decision logic
in a real-time pipeline with threading and buffering support.

Validates: Requirements 1.2, 5.4, 9.3
"""

import time
import threading
import queue
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..face_detection.face_detector import FaceDetector, FaceDetectionResult
from ..face_detection.landmark_detector import FacialLandmarkDetector
from ..feature_extraction.ear_calculator import EARCalculator
from ..feature_extraction.mar_calculator import MARCalculator
from ..ml_models.base_model import MLModel
from ..decision_logic.decision_engine import DecisionEngine, DrowsinessAssessment
from ..decision_logic.alert_manager import AlertManager, AlertLevel
from .camera_manager import CameraManager, FrameData


@dataclass
class DrowsinessIndicators:
    """Container for drowsiness indicators"""
    ear_value: float
    mar_value: float
    blink_rate: float
    yawn_detected: bool
    head_pose_score: float
    ml_confidence: float


class ProcessingStatus(Enum):
    """Frame processor status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ProcessingResult:
    """Result of frame processing"""
    timestamp: float
    frame_number: int
    face_detected: bool
    drowsiness_score: float
    alert_level: AlertLevel
    processing_time_ms: float
    confidence: float
    indicators: Optional[DrowsinessIndicators] = None
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for the processing pipeline"""
    total_frames_processed: int
    frames_with_face: int
    frames_without_face: int
    average_processing_time_ms: float
    average_fps: float
    peak_fps: float
    min_fps: float
    total_alerts_triggered: int
    face_detection_rate: float


class FrameProcessor:
    """
    Real-time frame processor for continuous drowsiness detection.
    
    Integrates all components in a threaded pipeline for real-time
    video analysis with performance monitoring and latency measurement.
    """
    
    def __init__(
        self,
        camera_manager: CameraManager,
        face_detector: FaceDetector,
        landmark_detector: FacialLandmarkDetector,
        ear_calculator: EARCalculator,
        mar_calculator: MARCalculator,
        ml_model: Optional[MLModel],
        decision_engine: DecisionEngine,
        alert_manager: AlertManager,
        buffer_size: int = 5
    ):
        """
        Initialize the FrameProcessor.
        
        Args:
            camera_manager: Camera manager for frame capture
            face_detector: Face detection component
            landmark_detector: Facial landmark detection component
            ear_calculator: Eye aspect ratio calculator
            mar_calculator: Mouth aspect ratio calculator
            ml_model: Optional ML model for drowsiness classification
            decision_engine: Decision logic engine
            alert_manager: Alert management system
            buffer_size: Size of frame buffer for threading
        """
        self.camera_manager = camera_manager
        self.face_detector = face_detector
        self.landmark_detector = landmark_detector
        self.ear_calculator = ear_calculator
        self.mar_calculator = mar_calculator
        self.ml_model = ml_model
        self.decision_engine = decision_engine
        self.alert_manager = alert_manager
        
        # Threading components
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.result_buffer = queue.Queue(maxsize=buffer_size)
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Status and metrics
        self.status = ProcessingStatus.STOPPED
        self.total_frames = 0
        self.frames_with_face = 0
        self.frames_without_face = 0
        self.total_processing_time = 0.0
        self.processing_times: List[float] = []
        self.fps_history: List[float] = []
        self.last_frame_time = 0.0
        
        # Callbacks
        self.on_result_callback: Optional[Callable[[ProcessingResult], None]] = None
        self.on_alert_callback: Optional[Callable[[AlertLevel, float], None]] = None
        self.on_error_callback: Optional[Callable[[str], None]] = None
    
    def start(self) -> bool:
        """
        Start the real-time processing pipeline.
        
        Returns:
            True if started successfully
        
        Validates: Requirements 1.2, 5.4
        """
        if self.status == ProcessingStatus.RUNNING:
            return True
        
        self.status = ProcessingStatus.STARTING
        
        # Ensure camera is ready
        if not self.camera_manager.isReady():
            success, message = self.camera_manager.initializeCamera()
            if not success:
                self.status = ProcessingStatus.ERROR
                if self.on_error_callback:
                    self.on_error_callback(f"Camera initialization failed: {message}")
                return False
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        self.status = ProcessingStatus.RUNNING
        return True
    
    def stop(self):
        """Stop the processing pipeline"""
        if self.status != ProcessingStatus.RUNNING:
            return
        
        self.stop_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.status = ProcessingStatus.STOPPED
    
    def pause(self):
        """Pause processing"""
        if self.status == ProcessingStatus.RUNNING:
            self.status = ProcessingStatus.PAUSED
    
    def resume(self):
        """Resume processing"""
        if self.status == ProcessingStatus.PAUSED:
            self.status = ProcessingStatus.RUNNING
    
    def processFrame(self, frame_data: FrameData) -> ProcessingResult:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame_data: Frame data to process
        
        Returns:
            ProcessingResult containing analysis results
        
        Validates: Requirements 1.2, 5.4, 9.3
        """
        start_time = time.time()
        
        try:
            # Step 1: Face Detection
            face_result = self.face_detector.detectFace(frame_data.frame)
            
            if not face_result.face_detected:
                self.frames_without_face += 1
                processing_time = (time.time() - start_time) * 1000
                
                return ProcessingResult(
                    timestamp=frame_data.timestamp,
                    frame_number=frame_data.frame_number,
                    face_detected=False,
                    drowsiness_score=0.0,
                    alert_level=AlertLevel.NONE,
                    processing_time_ms=processing_time,
                    confidence=0.0,
                    error="No face detected"
                )
            
            self.frames_with_face += 1
            
            # Step 2: Landmark Detection
            # Extract face region
            x, y, w, h = face_result.bounding_box
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame_data.frame.shape[1], x + w)
            y2 = min(frame_data.frame.shape[0], y + h)
            face_region = frame_data.frame[y:y2, x:x2]
            
            landmarks = self.landmark_detector.extractLandmarks(face_region)
            
            if landmarks is None or len(landmarks.landmarks) == 0:
                processing_time = (time.time() - start_time) * 1000
                return ProcessingResult(
                    timestamp=frame_data.timestamp,
                    frame_number=frame_data.frame_number,
                    face_detected=True,
                    drowsiness_score=0.0,
                    alert_level=AlertLevel.NONE,
                    processing_time_ms=processing_time,
                    confidence=0.0,
                    error="Landmark detection failed"
                )
            
            # Step 3: Feature Extraction
            left_ear = self.ear_calculator.calculateEAR(landmarks.get_left_eye())
            right_ear = self.ear_calculator.calculateEAR(landmarks.get_right_eye())
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = self.mar_calculator.calculateMAR(landmarks.get_mouth())
            
            # Update blink and yawn detection
            self.ear_calculator.updateEARHistory(avg_ear)
            self.mar_calculator.updateMARHistory(mar)
            
            blink_detected = self.ear_calculator.detectBlink()
            yawn_detected = self.mar_calculator.detectYawn()
            
            # Step 4: ML Inference (if model available)
            ml_confidence = 0.5  # Default neutral confidence
            if self.ml_model:
                try:
                    # Prepare input for ML model
                    face_region = self._extractFaceRegion(
                        frame_data.frame,
                        face_result.bounding_box
                    )
                    ml_result = self.ml_model.predict(face_region)
                    ml_confidence = ml_result.confidence
                except Exception as e:
                    print(f"ML inference error: {e}")
            
            # Step 5: Decision Logic
            indicators = DrowsinessIndicators(
                ear_value=avg_ear,
                mar_value=mar,
                blink_rate=self.ear_calculator.getBlinkRate(),
                yawn_detected=yawn_detected,
                head_pose_score=0.5,  # Placeholder for head pose
                ml_confidence=ml_confidence
            )
            
            # Convert indicators to scores for decision engine
            ear_score = 1.0 - avg_ear if avg_ear < 0.25 else 0.0  # Low EAR = drowsy
            mar_score = 1.0 if yawn_detected else 0.0
            head_pose_score = 0.5  # Placeholder
            
            assessment = self.decision_engine.calculate_drowsiness_score(
                ear_score=ear_score,
                mar_score=mar_score,
                head_pose_score=head_pose_score,
                ml_confidence=ml_confidence,
                timestamp=frame_data.timestamp
            )
            
            drowsiness_score = assessment.drowsiness_score
            alert_level = assessment.alert_level
            confidence = assessment.confidence
            
            # Step 6: Alert Management
            if alert_level != AlertLevel.NONE:
                self.alert_manager.triggerAlert(alert_level, drowsiness_score)
                if self.on_alert_callback:
                    self.on_alert_callback(alert_level, drowsiness_score)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            self.processing_times.append(processing_time)
            
            # Keep only recent processing times (last 100)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            return ProcessingResult(
                timestamp=frame_data.timestamp,
                frame_number=frame_data.frame_number,
                face_detected=True,
                drowsiness_score=drowsiness_score,
                alert_level=alert_level,
                processing_time_ms=processing_time,
                confidence=confidence,
                indicators=indicators
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            if self.on_error_callback:
                self.on_error_callback(f"Frame processing error: {str(e)}")
            
            return ProcessingResult(
                timestamp=frame_data.timestamp,
                frame_number=frame_data.frame_number,
                face_detected=False,
                drowsiness_score=0.0,
                alert_level=AlertLevel.NONE,
                processing_time_ms=processing_time,
                confidence=0.0,
                error=str(e)
            )
    
    def getPerformanceMetrics(self) -> PerformanceMetrics:
        """
        Get performance metrics for the processing pipeline.
        
        Returns:
            PerformanceMetrics object with statistics
        
        Validates: Requirements 9.3
        """
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        avg_fps = (
            sum(self.fps_history) / len(self.fps_history)
            if self.fps_history else 0.0
        )
        
        peak_fps = max(self.fps_history) if self.fps_history else 0.0
        min_fps = min(self.fps_history) if self.fps_history else 0.0
        
        face_detection_rate = (
            self.frames_with_face / self.total_frames
            if self.total_frames > 0 else 0.0
        )
        
        return PerformanceMetrics(
            total_frames_processed=self.total_frames,
            frames_with_face=self.frames_with_face,
            frames_without_face=self.frames_without_face,
            average_processing_time_ms=avg_processing_time,
            average_fps=avg_fps,
            peak_fps=peak_fps,
            min_fps=min_fps,
            total_alerts_triggered=self.alert_manager.getTotalAlerts(),
            face_detection_rate=face_detection_rate
        )
    
    def setResultCallback(self, callback: Callable[[ProcessingResult], None]):
        """Set callback for processing results"""
        self.on_result_callback = callback
    
    def setAlertCallback(self, callback: Callable[[AlertLevel, float], None]):
        """Set callback for alerts"""
        self.on_alert_callback = callback
    
    def setErrorCallback(self, callback: Callable[[str], None]):
        """Set callback for errors"""
        self.on_error_callback = callback
    
    def reset(self):
        """Reset processor state and metrics"""
        self.total_frames = 0
        self.frames_with_face = 0
        self.frames_without_face = 0
        self.total_processing_time = 0.0
        self.processing_times.clear()
        self.fps_history.clear()
        self.last_frame_time = 0.0
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        while not self.stop_event.is_set():
            if self.status != ProcessingStatus.RUNNING:
                time.sleep(0.01)
                continue
            
            try:
                # Capture frame
                frame_data = self.camera_manager.captureFrame()
                
                if frame_data is None or not frame_data.is_valid():
                    continue
                
                # Calculate FPS
                current_time = time.time()
                if self.last_frame_time > 0:
                    frame_interval = current_time - self.last_frame_time
                    current_fps = 1.0 / frame_interval if frame_interval > 0 else 0.0
                    self.fps_history.append(current_fps)
                    
                    # Keep only recent FPS values (last 100)
                    if len(self.fps_history) > 100:
                        self.fps_history.pop(0)
                
                self.last_frame_time = current_time
                
                # Process frame
                result = self.processFrame(frame_data)
                self.total_frames += 1
                
                # Call result callback
                if self.on_result_callback:
                    self.on_result_callback(result)
                
                # Add to result buffer (non-blocking)
                try:
                    self.result_buffer.put_nowait(result)
                except queue.Full:
                    # Remove oldest result if buffer is full
                    try:
                        self.result_buffer.get_nowait()
                        self.result_buffer.put_nowait(result)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                if self.on_error_callback:
                    self.on_error_callback(f"Processing loop error: {str(e)}")
                time.sleep(0.1)  # Brief pause on error
    
    def _extractFaceRegion(
        self,
        frame: np.ndarray,
        bounding_box: tuple
    ) -> np.ndarray:
        """
        Extract face region from frame.
        
        Args:
            frame: Full frame
            bounding_box: (x, y, width, height) of face
        
        Returns:
            Cropped face region
        """
        x, y, w, h = bounding_box
        
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        return frame[y:y2, x:x2]
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
