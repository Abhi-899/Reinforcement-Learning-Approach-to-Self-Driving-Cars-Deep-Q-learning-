# Reinforcement-Learning-Approach-to-Self-Driving-Cars-Deep-Q-learning
In this project our aim is to ; to train the self-driving car to make round
trips between the left top end and right bottom. We try to achieve this by training a deep q-learning agent using an environment where we allow the agent perform a certain no. of actions(turn 0° (that is, move forward), turn 20° to the left, turn 20° to the right in this case study) based on a reward policy.


https://user-images.githubusercontent.com/64439578/128899114-fb08717e-c6a3-4fff-ae62-ecf05456b374.mp4


## Building the environment
The whole environment for this project is built with Kivy, from start to finish. The implementatipon is done in the car.kv file where we define the three sensors that helps us record the distance from the sand obstacles. And we communicate with the car through map.py.
![Screenshot (3)](https://user-images.githubusercontent.com/64439578/128899727-a30e2bb7-1b80-4d9f-b940-0ee254fb7f64.png)

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
## The input states
In the input state, we need information telling the AI whether it is about to move
off the road or hit an obstacle.Our car has three sensors giving us
signals about how much sand is around them.The blue sensor tells us if there's any
sand at the left of the car, the red sensor tells us if there is any sand in front of the
car, and the yellow sensor tells us if there is any sand at the right of the car. The
signals of these sensors are already coded into three variables: signal1, signal2,
and signal3. These signals will tell the AI if it's about to hit some obstacle or about
to get out of the road, since the road is delimited by sand.With these four
elements, signal1, signal2, signal3, and orientation, you have everything
you need to be able to drive from one location to another, while staying on the road,
and without hitting any obstacles.Input state = (orientation, signal1, signal2, signal3)

## The Output Actions
The three possible actions of move forward, turn left,
and turn right make logical sense with the goal, constraint, and input states we
have, and we can define them as the three following rotations:

rotations = [turn 0° (that is, move forward), turn 20° to the left, turn 20° to the right]

## The Rewards
We must simply remember what the goal and
constraints are:

• The goal is to make round trips between the Airport and Downtown.

• The constraints are to stay on the road and avoid obstacles if any. In other
words, the constraint is to stay away from the sand.

Let us have a look at the reward policy:

1. We give the AI a good reward when it gets closer to the destination.

2. We give the AI a bad reward when it gets further away from the destination.

3. We give the AI a bad reward if it's about to drive onto some sand.

On that note, here are the rewards we'll give in each case:

1. The AI gets a bad reward of -1 if it drives onto some sand. 

2. The AI gets a bad reward of -0.2 if it moves away from the destination.

3. The AI gets a good reward of 0.1 if it moves closer to the destination.

## The AI Solution:
The ai or brain of the car is defined using the following deep Q-learning process(in ai.py file):

Initialization:

1. The memory of the experience replay is initialized to an empty list, called
memory in the code.

2. The maximum size of the memory is set, called capacity in the code.

At each time t, the AI repeats the following process, until the end of the epoch:

1. The AI predicts the Q-values of the current state st. Therefore, since three
actions can be played (0 <-> 0°, 1 <-> 20°, or 2 <-> -20°), it gets three
predicted Q-values.

2. The AI performs an action selected by the Softmax method:
   
   a_t=Softmax{Q(s_t,a)}

3. The AI receives a reward R(s_t,a_t), which is one of -1, -0.2 or +0.1.

4. The AI reaches the next state s_(t+1), which is composed of the next three signals
from the three sensors, plus the orientation of the car.

5. The AI appends the transition (s_t,a_t,r_t,s_(t+1)) to the memory.

6. The AI takes a random batch B which is a subset of M of transitions. For all the transitions
(s_tB,a_tB,r_tB,s_(tB+1)) of the random batch B:

° The AI gets the predictions: Q(s_tB,a_tB)

° The AI gets the targets

° The AI computes the loss between the predictions and the targets over
the whole batch B

![Screenshot (4)](https://user-images.githubusercontent.com/64439578/128852019-eae1df3b-7548-4e53-b964-0d2df62fd6c2.png)

° Finally, the AI backpropagates this loss error into the neural network,
and through stochastic gradient descent updates the weights according
to how much they contributed to the loss error.

The entire implementation of the above steps are done in the 'ai.py' file. The model structure is as given below:

![Screenshot (7)](https://user-images.githubusercontent.com/64439578/128895033-0790b442-b8ef-46d3-9748-c8d7ea4031c4.png)

## Demo
Now all we have to do is install python 3.6, Kivy and pytorch. Then run the map.py file using command on the CMD:
```
python map.py
```
Once you click on the save button the states and the weigts optimization of the previous transition are saved in the last_brain.pth file and load button loads the model updates of the last transition based on which the car makes the movements. The plot for rewards vs iterations is given below:
![Rewards_Plot](https://user-images.githubusercontent.com/64439578/128896658-d6305607-e3c1-407a-a187-07e0f0eb1f04.png)

## Further Improvements
1. Add another hidden layer to the DQN also using RELU.

2. Once the car “insect” reaches the goal it earns a reward of 2

3. Decrease punishment from going further away from the goal

4. Set temperature parameter to 75.

5. Add a timer for how long it takes for the agent to reach the destination. If the agent does not find the destination after 10 seconds it gets a punishment (reward -= 0.3), after 20 seconds more punishment (reward -= 0.5) and so on. The more time it takes to find the destination the more punishment it gets. Added this timer to the list of signals passed to the DQN so it can learn from it:
```
last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation, self.last_time]
```
6.  I’m still trying to tune the rewards and maybe add some dimensionality to the Dqn in the future in order to improve the performance. I am also working on map that has a larger window so I can draw more complex paths. Maybe add Prioritized Experience Replay.










