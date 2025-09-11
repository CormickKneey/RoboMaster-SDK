# RoboMaster AI Engine

A long-running program that connects to RoboMaster robots, captures camera feed, and uses OpenAI for intelligent robot control.

## Features

- **Robot Connection**: Supports multiple connection types (WiFi direct, network, USB)
- **Camera Integration**: Real-time video stream processing
- **AI-Powered Control**: Uses OpenAI GPT-4 Vision for intelligent decision making
- **Multi-threaded Architecture**: Separate threads for camera capture, AI inference, and robot control
- **Robust Error Handling**: Graceful shutdown and error recovery
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Prerequisites

1. **RoboMaster Robot**: EP series or compatible model
2. **OpenAI API Key**: Required for AI inference
3. **Python Environment**: Python 3.7 or higher

## Installation

1. Install dependencies:
```bash
pip install -r engine_requirements.txt
```

2. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Configuration

The engine supports three connection modes:

### WiFi Network Mode (Recommended)
```python
config = {
    'conn_type': 'sta',  # Network connection
    'proto_type': 'tcp'  # TCP protocol
}
```

### WiFi Direct Mode
```python
config = {
    'conn_type': 'ap',   # Direct WiFi connection
    'proto_type': 'udp'  # UDP protocol (faster for real-time control)
}
```

### USB Connection Mode
```python
config = {
    'conn_type': 'rndis',  # USB RNDIS connection
    'proto_type': 'tcp'    # TCP protocol
}
```

## Usage

### Basic Usage
```bash
python engine.py
```

### Programmatic Usage
```python
from engine import RoboMasterAIEngine

# Create engine instance
engine = RoboMasterAIEngine(
    conn_type='sta',
    proto_type='tcp',
    openai_api_key='your-key'
)

# Start the engine
engine.run_forever()
```

## Architecture

The engine uses a multi-threaded architecture:

1. **Camera Capture Thread**: Continuously captures frames from robot camera
2. **AI Inference Thread**: Processes frames using OpenAI GPT-4 Vision
3. **Control Thread**: Executes robot control commands based on AI decisions
4. **Main Thread**: Coordinates all operations and handles shutdown

## AI Control Logic

The AI system analyzes camera images and makes decisions based on:

- **Obstacle Detection**: Avoids obstacles by turning or stopping
- **Path Planning**: Chooses optimal movement directions
- **Gimbal Control**: Adjusts camera angle for better visibility
- **Safety First**: Prioritizes safe operation over aggressive movement

### Available Actions

- `move_forward`: Move robot forward
- `move_backward`: Move robot backward
- `turn_left`: Turn robot left
- `turn_right`: Turn robot right
- `stop`: Stop all movement
- `gimbal_up/down/left/right`: Adjust camera angle

## Safety Features

- **Confidence Thresholding**: Only executes high-confidence AI decisions
- **Graceful Shutdown**: Handles Ctrl+C and system signals properly
- **Error Recovery**: Continues operation despite temporary failures
- **Connection Monitoring**: Automatically handles connection issues

## Troubleshooting

### Connection Issues
1. Ensure robot is powered on and in correct mode
2. Check network connectivity (for 'sta' mode)
3. Verify robot IP address and connection type

### OpenAI API Issues
1. Verify API key is set correctly
2. Check internet connectivity
3. Monitor API rate limits and usage

### Camera Issues
1. Ensure camera module is functioning
2. Check camera resolution settings
3. Verify sufficient bandwidth for video stream

## Logs

The engine creates detailed logs in:
- Console output (real-time)
- Log file: `robomaster_engine_YYYYMMDD_HHMMSS.log`

## Performance Tuning

### For Real-time Control
- Use `conn_type='ap'` and `proto_type='udp'`
- Reduce camera resolution if needed
- Adjust inference frequency in code

### For Stability
- Use `conn_type='sta'` and `proto_type='tcp'`
- Increase error handling timeouts
- Enable more verbose logging

## Development

### Adding New Actions
1. Add action to `_perform_inference()` prompt
2. Implement action in `_execute_control_command()`
3. Test with robot in safe environment

### Customizing AI Behavior
- Modify the prompt in `_perform_inference()`
- Adjust confidence thresholds
- Add custom image preprocessing

## License

This project follows the same license as the RoboMaster SDK (Apache License 2.0).

## Safety Warning

⚠️ **Always operate the robot in a safe environment with adequate space and supervision. The AI system is experimental and may make unexpected decisions.**
