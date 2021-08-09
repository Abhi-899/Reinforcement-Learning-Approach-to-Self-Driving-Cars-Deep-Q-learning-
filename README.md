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

We have got to define how these signals are computed. We start by introducing a new variable, called sand, which
we initialize as an array that has as many cells as our graphic interface has pixels.
Simply put, the sand array is the black map itself and the pixels are the cells of the
array. Then, each cell of the sand array will get a 1 if there is sand, and a 0 if there
is not.
For example, here the sand array has only 1s in its first few rows, and the rest
is all 0s.Since the sand array only contains 1s (where there's sand) and 0s (where there's no
sand), we can very easily count the number of 1s by simply summing the cells of the
sand array in this 20 by 20 square. That gives us exactly the density of sand around
each sensor, and that's what's computed in the map.py file:
```
self.signal1 = int(np.sum(sand[int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

```
Let us now look into the input state vector:

1. goal_x: The x-coordinate of the goal (which can either be the left top or the right bottom)

2. goal_y: The y-coordinate of the goal (which can either be the left top or the right bottom)

3. xx = (goal_x - self.car.x): The difference of x-coordinates between the goal
and the car

4. yy = (goal_y - self.car.y): The difference of y-coordinates between the goal
and the car

5. orientation: The angle that measures the direction of the car with respect
to the goal

 We need to know how orientation is computed;
it's the angle between the axis of the car (the velocity vector from our first list
of parameters) and the axis that joins the goal and the center of the car. The goal has
the coordinates (goal_x, goal_y) and the center of the car has the coordinates (self.
car.x, self.car.y). For example, if the car is heading perfectly toward the goal,
then orientation = 0°. If you're curious as to how we can compute the angle between
the two axes in Python, here's the code that gets the orientation:
```
xx = goal_x - self.car.x
yy = goal_y - self.car.y
orientation = Vector(*self.car.velocity).
angle((xx,yy))/180.

```

