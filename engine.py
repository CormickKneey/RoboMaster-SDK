#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboMaster AI Engine - Long Running Program
Connects to RoboMaster robot, captures camera feed, and uses OpenAI for inference to control robot
"""

import os
import sys
import time
import logging
import threading
import queue
import signal
import base64
from typing import Optional, Dict, Any
from datetime import datetime

import cv2
import numpy as np
from openai import OpenAI
from robomaster import robot, camera


class RoboMasterAIEngine:
    """
    Main engine class for RoboMaster AI control system
    Integrates robot connection, camera capture, OpenAI inference, and robot control
    """
    
    def __init__(self, 
                 conn_type: str = "sta", 
                 proto_type: str = "tcp",
                 openai_api_key: Optional[str] = None,
                 openai_base_url: Optional[str] = None,
                 camera_resolution: str = camera.STREAM_720P):
        """
        Initialize the AI engine
        
        Args:
            conn_type: Connection type ('sta', 'ap', 'rndis')
            proto_type: Protocol type ('tcp', 'udp') 
            openai_api_key: OpenAI API key (if None, will try to get from env)
            camera_resolution: Camera resolution setting
        """
        self.conn_type = conn_type
        self.proto_type = proto_type
        self.camera_resolution = camera_resolution
        
        # Initialize components
        self.robot_instance: Optional[robot.Robot] = None
        self.openai_client: Optional[OpenAI] = None
        
        # Control flags
        self.running = False
        self.camera_active = False
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=10)
        self.inference_queue = queue.Queue(maxsize=5)
        self.control_queue = queue.Queue(maxsize=20)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize OpenAI client
        self._init_openai_client(openai_api_key, openai_base_url)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'robomaster_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _init_openai_client(self, api_key: Optional[str], base_url: Optional[str]):
        """Initialize OpenAI client"""
        try:
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                self.logger.warning("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
                return
                
            self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
            self.logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        
    def connect_robot(self) -> bool:
        """
        Connect to RoboMaster robot
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to robot with conn_type={self.conn_type}, proto_type={self.proto_type}")
            
            self.robot_instance = robot.Robot()
            self.robot_instance.initialize(conn_type=self.conn_type, proto_type=self.proto_type)
            
            # Get robot version to verify connection
            version = self.robot_instance.get_version()
            self.logger.info(f"Successfully connected to robot. Version: {version}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to robot: {e}")
            return False
            
    def disconnect_robot(self):
        """Disconnect from robot safely"""
        try:
            if self.robot_instance:
                if self.camera_active:
                    self.robot_instance.camera.stop_video_stream()
                    self.camera_active = False
                    
                self.robot_instance.close()
                self.robot_instance = None
                self.logger.info("Robot disconnected successfully")
                
        except Exception as e:
            self.logger.error(f"Error during robot disconnection: {e}")
            
    def start_camera_stream(self) -> bool:
        """
        Start camera video stream
        
        Returns:
            bool: True if stream started successfully, False otherwise
        """
        try:
            if not self.robot_instance:
                self.logger.error("Robot not connected. Cannot start camera stream.")
                return False
                
            camera_module = self.robot_instance.camera
            camera_module.start_video_stream(display=False, resolution=self.camera_resolution)
            self.camera_active = True
            
            self.logger.info("Camera stream started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start camera stream: {e}")
            return False
            
    def capture_frame_worker(self):
        """Worker thread for capturing camera frames"""
        self.logger.info("Camera capture worker started")
        
        while self.running and self.camera_active:
            try:
                if self.robot_instance and self.robot_instance.camera:
                    frame = self.robot_instance.camera.read_cv2_image()
                    
                    if frame is not None:
                        # Add timestamp to frame
                        timestamp = time.time()
                        
                        # Put frame in queue (non-blocking)
                        try:
                            self.frame_queue.put_nowait((timestamp, frame))
                        except queue.Full:
                            # Remove oldest frame if queue is full
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait((timestamp, frame))
                            except queue.Empty:
                                pass
                                
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                self.logger.error(f"Error in camera capture: {e}")
                time.sleep(1)
                
        self.logger.info("Camera capture worker stopped")
        
    def inference_worker(self):
        """Worker thread for OpenAI inference"""
        self.logger.info("Inference worker started")
        
        while self.running:
            try:
                # Get latest frame (blocking with timeout)
                timestamp, frame = self.frame_queue.get(timeout=1.0)
                
                # Skip inference if OpenAI client not available
                if not self.openai_client:
                    continue
                    
                # Perform inference every few frames to avoid overload
                if int(timestamp * 10) % 10 == 0:  # Process every 1 second approximately
                    result = self._perform_inference(frame)
                    
                    if result:
                        # Put inference result in control queue
                        try:
                            self.control_queue.put_nowait((timestamp, result))
                        except queue.Full:
                            # Remove oldest control command if queue is full
                            try:
                                self.control_queue.get_nowait()
                                self.control_queue.put_nowait((timestamp, result))
                            except queue.Empty:
                                pass
                                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in inference worker: {e}")
                time.sleep(1)
                
        self.logger.info("Inference worker stopped")
        
    def _perform_inference(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Perform OpenAI inference on camera frame
        
        Args:
            frame: Camera frame as numpy array
            
        Returns:
            Dict containing inference results and control commands
        """
        try:
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create prompt for robot control
            prompt = """
            You are controlling a RoboMaster robot. Analyze this camera image and provide control commands.
            
            Available actions:
            - move_forward: Move robot forward
            - move_backward: Move robot backward  
            - turn_left: Turn robot left
            - turn_right: Turn robot right
            - stop: Stop robot movement
            - gimbal_up: Move gimbal up
            - gimbal_down: Move gimbal down
            - gimbal_left: Move gimbal left
            - gimbal_right: Move gimbal right
            
            Respond with JSON format:
            {
                "action": "action_name",
                "description": "Brief description of what you see and why you chose this action",
                "confidence": 0.8
            }
            
            If you see obstacles ahead, choose to turn or stop. If path is clear, you can move forward.
            """
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to extract JSON from response
            import json
            try:
                # Look for JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    result = json.loads(json_str)
                    self.logger.info(f"Inference result: {result}")
                    return result
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse JSON from response: {content}")
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI inference: {e}")
            return None
            
    def control_worker(self):
        """Worker thread for robot control based on inference results"""
        self.logger.info("Control worker started")
        
        while self.running:
            try:
                # Get control command (blocking with timeout)
                timestamp, inference_result = self.control_queue.get(timeout=1.0)
                
                if not self.robot_instance:
                    continue
                    
                # Execute control command
                self._execute_control_command(inference_result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in control worker: {e}")
                time.sleep(1)
                
        self.logger.info("Control worker stopped")
        
    def _execute_control_command(self, inference_result: Dict[str, Any]):
        """
        Execute control command on robot
        
        Args:
            inference_result: Dictionary containing action and parameters
        """
        try:
            action = inference_result.get('action', 'stop')
            description = inference_result.get('description', '')
            confidence = inference_result.get('confidence', 0.0)
            
            self.logger.info(f"Executing action: {action} (confidence: {confidence}) - {description}")
            
            # Only execute if confidence is above threshold
            if confidence < 0.5:
                self.logger.info("Low confidence, skipping action")
                return
                
            chassis = self.robot_instance.chassis
            gimbal = self.robot_instance.gimbal
            
            # Execute movement commands
            if action == 'move_forward':
                chassis.move(x=0.3, y=0, z=0, xy_speed=0.5)
            elif action == 'move_backward':
                chassis.move(x=-0.3, y=0, z=0, xy_speed=0.5)
            elif action == 'turn_left':
                chassis.move(x=0, y=0, z=30, z_speed=45)
            elif action == 'turn_right':
                chassis.move(x=0, y=0, z=-30, z_speed=45)
            elif action == 'stop':
                chassis.drive_speed(x=0, y=0, z=0)
            elif action == 'gimbal_up':
                gimbal.move(pitch=10, yaw=0)
            elif action == 'gimbal_down':
                gimbal.move(pitch=-10, yaw=0)
            elif action == 'gimbal_left':
                gimbal.move(pitch=0, yaw=15)
            elif action == 'gimbal_right':
                gimbal.move(pitch=0, yaw=-15)
            else:
                self.logger.warning(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error executing control command: {e}")
            
    def start(self) -> bool:
        """
        Start the AI engine
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            self.logger.info("Starting RoboMaster AI Engine...")
            
            # Connect to robot
            if not self.connect_robot():
                return False
                
            # Start camera stream
            if not self.start_camera_stream():
                return False
                
            # Set running flag
            self.running = True
            
            # Start worker threads
            self.capture_thread = threading.Thread(target=self.capture_frame_worker, daemon=True)
            self.inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
            self.control_thread = threading.Thread(target=self.control_worker, daemon=True)
            
            self.capture_thread.start()
            self.inference_thread.start()
            self.control_thread.start()
            
            self.logger.info("RoboMaster AI Engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start AI engine: {e}")
            self.stop()
            return False
            
    def stop(self):
        """Stop the AI engine gracefully"""
        self.logger.info("Stopping RoboMaster AI Engine...")
        
        # Set stop flag
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        if hasattr(self, 'inference_thread') and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2)
        if hasattr(self, 'control_thread') and self.control_thread.is_alive():
            self.control_thread.join(timeout=2)
            
        # Disconnect robot
        self.disconnect_robot()
        
        self.logger.info("RoboMaster AI Engine stopped")
        
    def run_forever(self):
        """Run the engine indefinitely until interrupted"""
        if not self.start():
            self.logger.error("Failed to start engine")
            return
            
        try:
            self.logger.info("Engine running... Press Ctrl+C to stop")
            while self.running:
                time.sleep(1)
                
                # Print status every 30 seconds
                if int(time.time()) % 30 == 0:
                    self.logger.info(f"Engine status - Running: {self.running}, Camera: {self.camera_active}")
                    
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            self.stop()


def main():
    """Main function to run the AI engine"""
    # Configuration - modify these as needed
    config = {
        'conn_type': 'sta',  # 'sta' for network, 'ap' for direct WiFi, 'rndis' for USB
        'proto_type': 'tcp',  # 'tcp' or 'udp'
        'camera_resolution': camera.STREAM_720P,  # Camera resolution
        'openai_api_key': None,  # Will use OPENAI_API_KEY env var if None
        'openai_base_url': None  # Will use OPENAI_BASE_URL env var if None
    }
    
    # Create and run engine
    engine = RoboMasterAIEngine(**config)
    engine.run_forever()


if __name__ == '__main__':
    main()
