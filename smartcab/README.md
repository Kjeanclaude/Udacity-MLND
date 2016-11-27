# Machine Learning Engineer Nanodegree
# Reinforcement Learning
## Project 4 : Train a Smartcab How to Drive
-	Applied reinforcement learning to build a simulated vehicle navigation agent which runs safely and reliably.
-	Defined agent states based on US traffic laws at an intersection.
-	Defined a Q-learning algorithm from these states to help the Smartcab learn the US traffic laws.
-	Improved the Q-learning algorithm to reach the highest Safety and Reliability rating for the Smartcab (A+/A+).


### Install

This project requires **Python 2.7** with the [pygame](https://www.pygame.org/wiki/GettingStarted
) library installed

### Code

Template code is provided in the `smartcab/agent.py` python file. Additional supporting python code can be found in `smartcab/enviroment.py`, `smartcab/planner.py`, and `smartcab/simulator.py`. Supporting images for the graphical user interface can be found in the `images` folder. While some code has already been implemented to get you started, you will need to implement additional functionality for the `LearningAgent` class in `agent.py` when requested to successfully complete the project. 

### Run

In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains this README) and run one of the following commands:

```python smartcab/agent.py```  
```python -m smartcab.agent```

This will run the `agent.py` file and execute your agent code.
