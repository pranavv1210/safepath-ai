# SafePath AI - Detailed Masterplan

## 1. Project Title

SafePath AI: Intent and Trajectory Prediction with Risk-Aware Safety Analysis for Urban Autonomous Driving

## 2. Problem Statement Alignment

This project is built around the official problem statement:

In an L4 urban autonomous driving environment, reacting only to a pedestrian's current location is not sufficient. The system must anticipate where pedestrians and cyclists are likely to move in the near future. SafePath AI will therefore predict the next 3 seconds of future coordinates for vulnerable road users using the previous 2 seconds of observed motion.

The official challenge focuses on:

- Temporal sequence modeling
- Intent and trajectory prediction
- Multi-modal future prediction
- ADE and FDE evaluation
- Social context awareness
- nuScenes as the target dataset

SafePath AI is designed to satisfy those requirements directly, while also adding a practical innovation layer: collision risk analysis and an interactive visualization demo.

## 3. Project Vision

SafePath AI is an end-to-end trajectory prediction and safety interpretation system for urban autonomous driving scenarios. The core idea is to predict multiple possible future paths for pedestrians and cyclists, estimate the likelihood of each path, and identify which predicted futures are potentially dangerous for an autonomous vehicle.

The project combines:

- A high-accuracy trajectory prediction model
- A multi-modal prediction strategy that outputs the top 3 likely futures
- A risk scoring engine for safety interpretation
- A simple interactive dashboard for visualization and demo delivery

This makes the project not only a machine learning model, but also a complete demonstration of how prediction can be connected to decision support in autonomous systems.

## 4. Core Objective

The primary objective is to build a model that:

- Takes 2 seconds of past motion history as input
- Predicts 3 seconds of future trajectory as output
- Works for pedestrians and cyclists
- Generates multiple likely futures instead of a single deterministic path
- Minimizes ADE and FDE

The secondary objective is to build a risk-aware layer that:

- Evaluates the danger associated with each predicted trajectory
- Estimates whether a predicted path may intersect with the ego vehicle path
- Produces interpretable outputs such as risk level and collision likelihood

## 5. Why This Problem Matters

In dense urban environments, pedestrians and cyclists behave unpredictably. They may stop, accelerate, cross unexpectedly, avoid each other, or change direction. A self-driving vehicle must therefore predict not just a single future, but several likely futures, and understand which of those futures could become dangerous.

This problem matters because:

- Human motion is inherently uncertain
- Multiple valid futures can emerge from the same past trajectory
- Urban safety depends on early anticipation, not just fast reaction
- Prediction quality directly affects downstream planning and collision avoidance

## 6. Proposed System Overview

SafePath AI has two major layers.

### 6.1 Layer 1: Trajectory Prediction Model

This is the main challenge-solving component.

Input:

- Past 2 seconds of agent motion
- Coordinates: `(x, y)`
- Velocity: `(vx, vy)`
- Relative and normalized trajectory representation

Output:

- Predicted future `(x, y)` coordinates for the next 3 seconds
- Top 3 likely future trajectories
- Probability score for each predicted future

Primary optimization goals:

- ADE (Average Displacement Error)
- FDE (Final Displacement Error)

### 6.2 Layer 2: Risk Scoring Engine

This is the innovation layer added on top of the core challenge.

Input:

- Predicted trajectories from the model
- Probability of each predicted trajectory
- Ego vehicle reference path or simplified straight-line motion assumption

Output:

- Risk level: High / Medium / Low
- Estimated collision probability
- Safety explanation based on distance, path overlap, and time-to-collision

This layer helps turn trajectory prediction into actionable safety insight.

## 7. Scope of the First Version

The first version of SafePath AI will prioritize a strong baseline that is realistic to implement and present well.

Included in version 1:

- Sequence preprocessing pipeline
- Velocity-enhanced input features
- LSTM encoder-decoder trajectory model
- Multi-modal prediction with top 3 futures
- ADE and FDE evaluation
- Basic risk engine
- Flask inference API
- HTML/JavaScript visualization dashboard

Deferred to later versions:

- Transformer-based trajectory modeling
- Social pooling or graph-based interaction modeling
- Map-aware prediction
- Real-time streaming deployment
- Full planning integration

## 8. Dataset Strategy

### 8.1 Official Target Dataset

The target dataset for the official challenge is nuScenes.

This dataset is expected to provide:

- Urban driving scenes
- Pedestrian and cyclist trajectories
- Temporal observations across frames
- Real-world traffic context

### 8.2 Interim Development Strategy

Since the challenge dataset may be released later, development can begin using substitute public pedestrian trajectory datasets for pipeline validation.

Interim datasets:

- Stanford Drone Dataset
- ETH / UCY pedestrian datasets

Purpose of interim datasets:

- Validate preprocessing logic
- Train and debug the baseline model
- Test sequence generation and visualization flow
- Prepare the codebase so it can be adapted quickly once nuScenes is available

### 8.3 Data Preparation Tasks

The preprocessing pipeline will perform:

- Agent track extraction
- Past/future sequence generation
- Coordinate normalization
- Relative position conversion
- Velocity computation from coordinates
- Train/validation/test split
- Filtering of pedestrians and cyclists

## 9. Input and Output Design

### 9.1 Model Input

For each tracked pedestrian or cyclist:

- Past 2 seconds of motion history
- Sequence of coordinates `(x, y)`
- Derived velocity `(vx, vy)`
- Relative positions centered on the latest observed point or initial point

Possible input tensor structure:

- Sequence length x feature dimension
- Example features per timestep: `[x, y, vx, vy]`

### 9.2 Model Output

The model will predict:

- Future sequence of `(x, y)` coordinates over the next 3 seconds
- Three candidate future trajectories
- A confidence or probability score for each trajectory

## 10. Model Architecture Plan

### 10.1 Baseline Architecture

The baseline model will use an LSTM encoder-decoder architecture.

Encoder:

- Reads the past motion sequence
- Encodes motion dynamics into a hidden state

Decoder:

- Generates future coordinates step by step
- Outputs predicted future positions

Why this baseline is a good fit:

- Strong for temporal sequence data
- Easier to train and debug than more complex architectures
- Directly aligned with the problem statement focus areas
- Good foundation for future upgrades

### 10.2 Feature Design

The initial feature set will include:

- Position `(x, y)`
- Velocity `(vx, vy)`
- Relative displacement

Potential later additions:

- Acceleration
- Heading direction
- Neighbor proximity features
- Scene context

### 10.3 Multi-Modal Prediction Strategy

The challenge expects multiple possible future paths. To support this, SafePath AI will produce the top 3 likely trajectories.

Possible implementation strategies:

- Multiple decoder passes with stochastic dropout sampling
- A multi-head decoder that outputs several futures
- Best-of-K generation during training or inference

Initial practical strategy:

- Use repeated decoder sampling or multiple prediction heads
- Score each predicted path
- Return the 3 most likely futures

### 10.4 Social Context Extension

The official statement highlights social context, meaning agents do not move independently. People often adjust paths based on nearby pedestrians and cyclists.

For the first version:

- Social context will be acknowledged in design
- The baseline may use only single-agent history for simplicity

Planned upgrade:

- Add social pooling, local interaction encoding, or graph-based neighboring-agent features

This will strengthen alignment with the challenge once the baseline is stable.

### 10.5 Future Architecture Expansion

After the baseline is working, the following upgrades can be explored:

- GRU variant for lighter sequence modeling
- Transformer encoder-decoder for richer temporal learning
- Goal-conditioned prediction for intent-aware futures
- Social pooling or graph neural modules for interaction modeling

## 11. Loss Function and Training Strategy

### 11.1 Primary Loss

The baseline training loss will be Mean Squared Error on predicted future coordinates.

### 11.2 Enhanced Loss Design

To improve final-position accuracy, the loss can be extended with:

- Higher weight on later forecast timesteps
- Final-point weighted penalty to improve FDE
- Multi-modal best-match loss if multiple futures are generated

### 11.3 Training Goals

The model should learn:

- Smooth future motion
- Realistic path continuation
- Low average displacement error
- Low final position error
- Stable multi-modal outputs

## 12. Evaluation Plan

### 12.1 Primary Metrics

The official evaluation metrics are:

- ADE: Mean Euclidean distance between predicted and true future points
- FDE: Euclidean distance between the final predicted point and actual final point

### 12.2 Additional Validation Checks

Beyond metrics, the model should also be checked for:

- Visual realism of predicted paths
- Stability across scenes
- Diversity of predicted futures
- Calibration of prediction probabilities

### 12.3 Success Target for Round 1

Round 1 success means:

- A working model that reliably predicts future trajectories
- Measurable ADE and FDE values
- Multi-modal output for 3 candidate futures
- A clear demonstration pipeline that can be shown in a presentation

## 13. Risk Scoring Engine Design

Although the official challenge focuses mainly on prediction, SafePath AI adds a risk analysis module to strengthen the practical value of the system.

### 13.1 Purpose

The risk engine converts raw predicted trajectories into safety signals.

### 13.2 Inputs

- Predicted future trajectories
- Probability of each trajectory
- Ego vehicle path or assumed forward motion

### 13.3 Core Computations

The risk engine will estimate:

- Minimum distance between predicted agent path and vehicle path
- Path intersection possibility
- Time-to-collision
- Confidence-weighted danger score

### 13.4 Outputs

- Risk category: High / Medium / Low
- Collision probability estimate
- Highlight of the most dangerous predicted path

### 13.5 Why It Matters

This component distinguishes SafePath AI from a standard forecasting model by showing how predictions can support downstream decision-making.

## 14. Visualization Dashboard

The system will include a lightweight demo dashboard to make the model easier to understand and present.

### 14.1 Main Dashboard Elements

Left panel:

- Past observed trajectory
- Ground truth future trajectory
- Predicted top 3 future trajectories
- Ego vehicle reference path

Right panel:

- ADE and FDE values for the current sample or run
- Risk levels for each predicted path
- Probability scores for each candidate future
- Scenario details

### 14.2 Visual Encoding

- Blue: past trajectory
- Green: low-risk prediction
- Yellow: medium-risk prediction
- Red: high-risk prediction
- Dashed or lighter styling for lower-probability paths

### 14.3 Interaction Design

- Frame-by-frame playback
- Sample selection
- Scenario replay
- Hover or click to inspect each predicted path

## 15. Technical Stack

### 15.1 Machine Learning

- Python
- PyTorch

### 15.2 Data Processing

- NumPy
- Pandas
- Scikit-learn

### 15.3 Backend

- Flask for inference APIs and integration

### 15.4 Frontend

- HTML
- JavaScript
- Canvas or SVG-based trajectory rendering

### 15.5 Experimentation

- Jupyter notebooks for exploration and debugging

## 16. Repository Structure

Recommended project structure:

```text
project/
|-- data/
|-- preprocessing/
|-- models/
|-- training/
|-- inference/
|-- risk_engine/
|-- visualization/
|-- app/
|-- notebooks/
|-- tests/
|-- README.md
|-- MASTERPLAN.md
```

Suggested directory roles:

- `data/`: raw and processed data references
- `preprocessing/`: sequence generation and feature engineering
- `models/`: LSTM, GRU, Transformer, and related modules
- `training/`: training loops, evaluation scripts, checkpoints
- `inference/`: model loading and prediction utilities
- `risk_engine/`: collision and risk logic
- `visualization/`: plotting and dashboard helpers
- `app/`: Flask app and UI integration
- `tests/`: unit tests for preprocessing, metrics, and inference

## 17. Development Roadmap

### Phase 1: Problem Framing and Dataset Pipeline

Goals:

- Finalize target task definition
- Prepare trajectory sequence extraction pipeline
- Support interim datasets until nuScenes is released

Deliverables:

- Clean preprocessing scripts
- Input-output tensor generation
- Velocity computation
- Normalization utilities

### Phase 2: Baseline Trajectory Model

Goals:

- Implement LSTM encoder-decoder
- Train on prepared sequence data
- Produce future coordinate predictions

Deliverables:

- Baseline trained model
- ADE/FDE evaluation output
- Saved checkpoints

### Phase 3: Multi-Modal Prediction

Goals:

- Extend baseline to produce 3 likely futures
- Assign confidence scores

Deliverables:

- Top 3 future trajectories
- Probability scoring logic
- Multi-modal inference output

### Phase 4: Risk Engine

Goals:

- Evaluate predicted futures against vehicle path assumptions
- Convert forecasts into danger scores

Deliverables:

- Risk scoring module
- Collision logic
- Risk label generation

### Phase 5: Dashboard and API

Goals:

- Expose inference through Flask
- Build browser-based visualization

Deliverables:

- Inference endpoint
- Demo UI
- Path and risk visualization

### Phase 6: Integration and Presentation Readiness

Goals:

- Connect model, risk engine, and dashboard
- Validate workflow end to end
- Prepare project for demo and submission

Deliverables:

- Working pipeline
- Screenshots and visual outputs
- Clean README and presentation assets

## 18. Round 1 Build Priorities

To stay focused, Round 1 should prioritize the following in order:

1. Build a clean preprocessing pipeline
2. Implement a reliable LSTM baseline
3. Measure ADE and FDE properly
4. Add top 3 multi-modal outputs
5. Add a simple but clear risk engine
6. Create a polished demo dashboard

This order ensures the required challenge objective is solved first, and innovation is layered on top after the baseline works.

## 19. Differentiation Strategy

SafePath AI stands out because it is not just a forecasting model. It presents a complete story:

- Strong baseline trajectory prediction
- Velocity-enhanced temporal modeling
- Multi-modal future generation
- Risk-aware interpretation
- Interactive visualization
- End-to-end architecture from data to demo

This makes the project stronger technically and easier to explain in judging, presentations, and GitHub documentation.

## 20. Risks and Mitigation

### Risk 1: Dataset availability delay

Mitigation:

- Use SDD or ETH/UCY first
- Keep preprocessing modular so it can be adapted quickly to nuScenes

### Risk 2: Multi-modal output instability

Mitigation:

- Start with a stable single-path model
- Extend gradually to top 3 prediction heads or stochastic decoding

### Risk 3: Social context complexity

Mitigation:

- Document it as a planned upgrade
- Keep baseline simple and working first

### Risk 4: Demo integration overhead

Mitigation:

- Use a lightweight Flask plus HTML/JS architecture
- Focus on clarity over visual complexity

## 21. Future Scope

After the first complete version, SafePath AI can be expanded in several directions:

- Social pooling for multi-agent interaction
- Transformer-based trajectory forecasting
- Goal-conditioned prediction for intent modeling
- Map and road context integration
- Real-time deployment and planning coupling
- Uncertainty calibration improvements
- Better risk estimation under dense traffic conditions

## 22. Success Criteria

The project will be considered successful if it achieves the following:

- A working end-to-end trajectory prediction pipeline
- Input of past motion and output of future coordinates
- Multi-modal top 3 trajectory prediction
- Reasonable ADE and FDE performance
- Clean visualization of predictions
- Basic but meaningful risk scoring
- Strong project explanation and documentation

## 23. PPT / Demo Storyline

Recommended presentation flow:

1. Urban autonomy needs prediction, not just perception
2. Pedestrian and cyclist motion is uncertain and multi-modal
3. SafePath AI predicts the next 3 seconds from the previous 2 seconds
4. The model outputs the 3 most likely future paths
5. ADE and FDE evaluate prediction quality
6. A risk engine identifies dangerous futures
7. The dashboard visualizes motion, probabilities, and safety levels
8. Future upgrades add social pooling, map context, and Transformers

## 24. Final Statement

SafePath AI is a trajectory forecasting system built for the official intent and trajectory prediction challenge. It directly addresses the required task of predicting 3 seconds of future pedestrian and cyclist motion from 2 seconds of past observations, while extending the solution with multi-modal forecasting, interpretable safety scoring, and an end-to-end demo pipeline.

The result is a project that is technically aligned with the challenge, practical to implement in stages, strong for presentation, and expandable into a more advanced autonomous behavior prediction system.
