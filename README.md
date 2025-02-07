# Vehicle Motion Prediction: A Comparative Study of Data-Driven and Physics-Based Models

In various fields such as autonomous driving, urban planning, and safety systems, accurate motion prediction for multiple objects, including vehicles and vulnerable road users, is critical. This project addresses the challenge of selecting the best motion prediction models for different entities, particularly car objects. The study compares data-driven and physics-based methodologies, evaluating which approach best captures the complex dynamics and behaviors of car objects.

By using the inD dataset, we empirically assess the performance of both models using Average Displacement Error (ADE) as the performance metric. The study reveals that data-driven models, leveraging machine learning techniques, are highly compatible with car motion prediction tasks, while physics-based models are more effective in scenarios with shorter time horizons.

## Installation
Prerequisites -
Ensure you have the following dependencies installed:

Python 3.x
pip (Python package manager)
Install Dependencies. Clone the repository and install the required Python packages.

## Usage
To run the motion prediction models:

1. Dataset: Ensure the inD dataset is available in the *data/* folder.

2. Evaluation: The models will output predictions, and performance will be evaluated using Average Displacement Error (ADE).

## Results
The study evaluates the models using Average Displacement Error (ADE) to compare how accurately each approach predicts the motion of car objects. Key findings include:

Data-Driven Models: These models performed well in capturing the complex dynamics of car objects, especially when trained with large datasets.
Physics-Based Models: These models were more effective in situations with shorter prediction horizons.
