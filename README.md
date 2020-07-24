# Mars rover project attempt by team_paradox
This project is rendered using HTML, CSS, Python, Javascript and React.  
## Introduction:
The scenario here optimizes the length of tour of rover using two algorithms implemented for shortest path searching.  

## 1.Travelling Salesman Problem:
##### Implemented using Python and embedded using Repl.it
## Aim: 
Given a set of locations which the rover needs to explore, the algorithm predicts the sequence of visiting the locations, which helps the rover to traverse all the locations in shortest time and with minimal length of tour.
## Implementation:
The user can interact with the subplot plotted using Matplotlib library in Python and feed in all the locations.
The steps for optimization include:

 ###### *Defining a fitness function object*:
 Specifying the (x,y) coordinates of the locations to be visited.The distance between the locations are calculated as the euclidean distances from the input coordinates.
  ###### *Defining an optimization problem object*:
  Once the fitness function is defined we use it to define a optimization problem.Since this is a minimisation problem,we set
  'maximize=False'.
  ###### *Select and run a randomized optimization algorithm*: 
   Here we select a genetic algorithm with the default parameter settings of a population size (pop_size) of 200, a mutation probability (mutation_prob) of 0.1, a maximum of 10 attempts per step (max_attempts) and no limit on the maximum total number of iteration of the algorithm (max_iters).
  
  ## How to run the project:
  1.Since this code is embedded into HTML using Repl.it,two tiles will be visible at any time
  \
  2.To start the console,click the run icon
  \
  3.Since these files have a lot of dependencies imported,wait for them to install and update
  \
  4.Once this is done a subplot will open, waiting for user's input
  \
  5.Go ahead and plot your destination locations and press 'enter' key using keyboard
  \
  6.The algorithm begins to find an optimal solution and returns a 'Sol' button on the bottom left corner
  \
  7.Click on the 'Sol' button and wait to view your results plotted in the sub plot and displayed on the console
  ## 2.Dijkstras Shortest Path Algorithm
##### Implemented using react, deployed as a GitHub page.
## Aim: 
Given a start position and an end position and  a number of obstacles, find the shortest path from the start node to the end node using Dijkstras shortest path algorithm.(no diagonals).
## Implementation:
The user interactions are enabled by using event listeners on html elements and using radio buttons.
  
  ## How to run the project:
  1.Enable option to choose the either start position or end position , double click anywhere on the grid to generate start position or end position based on the option enabled.
  \
  2.Enable the other option, double click anywhere on the grid to generate end or start position.
  \
  3.Click anywhere on grid to generate obstacles.
  \
  4.Visualise button become visible once the start and end positions are fixed.Click on the visualised button to see the visualisation.
  \
  5.Refresh page for trying out a new visualisation.
find visualisation of dijkstras at:https://lathikadevraj.github.io/shortestpath/
orginally deployed at :https://github.com/LathikaDevraj/shortestpath.

  
  
  
  
  
