# Reinforcement-Learning-Approach-to-Self-Driving-Cars-Deep-Q-learning
In this project our aim is to ; to train the self-driving car to make round
trips between the left top end and right bottom. We try to achieve this by training a deep q-learning agent using an environment where we allow the agent perform a certain no. of actions(turn 0° (that is, move forward), turn 20° to the left, turn 20° to the right in this case study) based on a reward policy.

## Building the environment
The whole environment for this project is built with Kivy, from start to finish. The implementatipon is done in the car.kv file where we define the three sensors that helps us record the distance from the sand obstacles. And we communicate with the car through map.py.

## Setting the Parameters
The inputs, outputs, and rewards are all functions of these parameters. Let's list them
all, using the same names as in the code, so that you can easily understand the file
map.py:

1. angle: The angle between the x-axis of the map and the axis of the car

2. rotation: The last rotation made by the car (we will see later that when
playing an action, the car makes a rotation)

3. pos = (self.car.x, self.car.y): The position of the car (self.car.x is the
x-coordinate of the car, self.car.y is the y-coordinate of the car)

4. velocity = (velocity_x, velocity_y): The velocity vector of the car

5. sensor1 = (sensor1_x, sensor1_y): The position of the first sensor

6. sensor2 = (sensor2_x, sensor2_y): The position of the second sensor

7. sensor3 = (sensor3_x, sensor3_y): The position of the third sensor

8. signal1: The signal received by sensor 1

9. signal2: The signal received by sensor 2

10. signal3: The signal received by sensor 3


