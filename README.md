# PromptServe

**PromptServe** is a voice-to-vision robotic serving prototype that uses a multi-agent VLM pipeline to understand what a user wants picked, identify the matching object in a tabletop scene, and trigger deterministic robot motion to move the object from a source box to a destination box.

The project is designed around a simple idea:

> Natural language should be enough to tell a robot what to serve.

## Project Vision

PromptServe explores how voice, vision, and robotic manipulation can work together in practical serving environments such as:

- Cafeteria lines
- Smart vending machines
- Food kiosks
- Assisted meal prep
- Accessibility-focused serving systems
- Retail or inventory pickup stations

A user can say something like:

```text
Pick the fruit.
Grab something sweet.
Move the toy.
Pick the burger and the fries.
```

The system interprets the request, identifies the matching object in the scene, validates the detection, and moves the selected item.

## Current Demo Setup

The current prototype uses a tabletop foam board with two marked boxes:

- **Source box**: contains three objects
- **Destination box**: where the selected object should be moved

The AgileX Piper arm starts from a neutral pose, moves to a camera capture pose, observes the board, and then performs the serving motion.

## Hardware

- **AgileX Piper robotic arm** for pick-and-place motion
- **Camera attached to the robotic arm** for tabletop image capture
- **Waveshare T5E1** for voice input
- **T5AI board** for status display
- **Foam board tabletop layout** with marked source and destination zones

## Software Architecture

PromptServe separates intelligence from motion control.

The physical robot motion is treated as a deterministic workflow. The AI system focuses on understanding what object should be picked.

```text
Voice command
   ↓
T5E1 captures user request
   ↓
Intent Agent parses the request
   ↓
Camera captures tabletop scene
   ↓
Detection Agent finds candidate object
   ↓
Validation Agent confirms the crop matches the intent
   ↓
Bounding box is converted to board / robot coordinates
   ↓
Deterministic Piper workflow executes pick and place
   ↓
T5AI displays status
```

## Agent Pipeline

The software uses the **Strands Agent framework** with three focused agents.

### 1. Intent Agent

Parses the user's request against the known menu or object list.

Example:

```text
"Grab something sweet"
→ target: strawberry
→ search query: "red strawberry on table"
```

For multi-step requests:

```text
"Pick the burger and the fries"
→ [burger, fries]
```

### 2. Detection Agent

Runs **Grounding DINO** with the optimized query from the Intent Agent.

Returns:

- Object label
- Bounding box
- Confidence score

### 3. Validation Agent

Looks at the cropped image from the detected bounding box and confirms whether it actually matches the user intent.

This catches cases where the detector is confident but wrong.

Example:

```text
Grounding DINO: "strawberry"
Actual crop: tomato
Validation Agent: reject and retry with a better query
```

## Robot Motion Workflow

The robot motion is designed to be deterministic and reliable.

```text
Workflow 0: Move from neutral pose to camera capture pose
Workflow 1: Identify requested object from image and text prompt
Workflow 2: Convert bounding box pixel location to robot target position
Workflow 3: Open gripper, descend, and grab object
Workflow 4: Move object to destination box
Workflow 5: Return arm to neutral resting position
```

## Current Status

The original plan was to perform dynamic object picking based on the detected bounding box. During implementation, we ran into multiple issues with live Piper arm operation and dynamic motion control.

The current working demo uses **recorded Piper arm motion playback** to keep the presentation reliable while still showcasing the core VLM understanding pipeline.

Current working / planned components:

- Tabletop serving layout
- VLM-based object understanding flow
- Strands-based multi-agent software design
- Grounding DINO detection step
- Validation agent concept
- T5E1 voice input path
- T5AI status display path
- Recorded Piper arm motion playback

## Why This Matters

Most robotic automation systems rely on fixed buttons, menus, or hardcoded item slots. PromptServe explores a more natural interface:

```text
User intent + visual scene → selected object → robotic action
```

The core technical bet is that VLM grounding can make robotic serving systems easier and more flexible to use.

## Known Limitations

- Dynamic live arm control is not fully working yet
- Current motion uses recorded Piper trajectories
- Pixel-to-robot coordinate calibration still needs to be stabilized
- Gripper timing and grasp reliability need improvement
- Object selection should remain constrained for reliable demos
- Full multi-item serving is planned but not complete

## Roadmap

- Complete dynamic Piper motion from detected object position
- Add robust pixel-to-board-to-robot calibration
- Improve gripper open / close timing
- Add safer motion planning around the table surface
- Fully integrate T5E1 voice input
- Display live workflow state on T5AI
- Support multi-object serving requests
- Add retry behavior when detection fails or validation rejects a crop

## Example Use Case

```text
User: "Pick something sweet."

Intent Agent:
  target = strawberry
  query = "red strawberry on table"

Detection Agent:
  bbox = [x1, y1, x2, y2]
  confidence = 0.86

Validation Agent:
  crop matches strawberry
  result = approved

Robot Workflow:
  move to object
  grab object
  place in destination box
```

## Project Positioning

PromptServe is a prototype of a larger idea: a serving robot that can understand natural language requests and ground them in the physical world.

The current demo is intentionally constrained, but the direction is clear:

> Agents decide what to pick. Deterministic robot motion handles how to pick it.

