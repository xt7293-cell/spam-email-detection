# Autonomous Maze Solver

This project trains an agent to solve a maze.

## Team Members

Chen Ziqian<br>
Joshua Cheng Si Le<br>
Nyi Nyi Soe San<br>
Wang Xiaotong<br>
Xu Zhekai<br>
Yao Jieyi

## Requirements

Autonomous maze solver: This project is to design an autonomous maze solver. The aim is to
train an RL agent to navigate through a maze and find the shortest path to the goal. The agent
receives rewards for reaching the goal and penalties for hitting walls or taking too many steps.
Students shall actively find additional information and resource to effectively conduct this
project. 

## Files

environment.py — Person 1: Manage maze rules, agent movement, rewards, and goal detection.

maze_generator.py — Person 2: Generate, save, and load solvable mazes of different sizes.

search.py — Person 3: Implement BFS and A* to find the shortest path through the maze.

q_learning.py — Person 4: Train a Q-learning agent and save or load its Q-table.

app.py — Person 5: Build the interface to select mazes, run algorithms, and animate solutions.

evaluate.py — Person 6: Run experiments and compare success rates, path lengths, runtime, and training progress.