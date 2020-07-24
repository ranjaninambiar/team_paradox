import matplotlib.pyplot as plt
import numpy as np
import opt_probs
from matplotlib.widgets import Button

fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111)
#ax.set_xlim([0, 10])
#ax.set_ylim([0, 10])
dpx=[]
dpy=[]
dp=[]
npl=[]
def onclick(event):
    global i
    global click_number
    
    plt.plot(event.xdata, event.ydata,marker='D' , markersize=2,color='red') 
    dp.append((event.xdata, event.ydata))
    dpx.append(event.xdata)
    dpy.append(event.ydata)
    # function to show the plot 
    ax.annotate(i, (event.xdata, event.ydata),xytext=(event.xdata+0.15, event.ydata+0.3),color='blue',size=14)
    #plt.plot(dpx,dpy)
    if(click_number==0):
    	click_number=1
    else:
    	plt.arrow(dpx[i-1],dpy[i-1],dpx[i]-dpx[i-1],dpy[i]-dpy[i-1],width=0.1,facecolor='green')
    i+=1
    plt.show()





def press(event):
    fig.canvas.mpl_disconnect(cid)
    myalgo()
def myalgo():
    class TravellingSales:


        def __init__(self, coords=None, distances=None):

            if coords is None and distances is None:
                raise Exception("""At least one of coords and distances must be"""
                                + """ specified.""")

            elif coords is not None:
                self.is_coords = True
                path_list = []
                dist_list = []

            else:
                self.is_coords = False

                # Remove any duplicates from list
                distances = list({tuple(sorted(dist[0:2]) + [dist[2]])
                                  for dist in distances})

                # Split into separate lists
                node1_list, node2_list, dist_list = zip(*distances)

                if min(dist_list) <= 0:
                    raise Exception("""The distance between each pair of nodes"""
                                    + """ must be greater than 0.""")
                if min(node1_list + node2_list) < 0:
                    raise Exception("""The minimum node value must be 0.""")

                if not max(node1_list + node2_list) == \
                        (len(set(node1_list + node2_list)) - 1):
                    raise Exception("""All nodes must appear at least once in"""
                                    + """ distances.""")

                path_list = list(zip(node1_list, node2_list))

            self.coords = coords
            self.distances = distances
            self.path_list = path_list
            self.dist_list = dist_list
            self.prob_type = 'tsp'

        def evaluate(self, state):
            """Evaluate the fitness of a state vector.

            Parameters
            ----------
            state: array
                State array for evaluation. Each integer between 0 and
                (len(state) - 1), inclusive must appear exactly once in the array.

            Returns
            -------
            fitness: float
                Value of fitness function. Returns :code:`np.inf` if travel between
                two consecutive nodes on the tour is not possible.
            """

            if self.is_coords and len(state) != len(self.coords):
                raise Exception("""state must have the same length as coords.""")

            if not len(state) == len(set(state)):
                raise Exception("""Each node must appear exactly once in state.""")

            if min(state) < 0:
                raise Exception("""All elements of state must be non-negative"""
                                + """ integers.""")

            if max(state) >= len(state):
                raise Exception("""All elements of state must be less than"""
                                + """ len(state).""")

            fitness = 0

            # Calculate length of each leg of journey
            for i in range(len(state) - 1):
                node1 = state[i]
                node2 = state[i + 1]

                if self.is_coords:
                    fitness += np.linalg.norm(np.array(self.coords[node1])
                                              - np.array(self.coords[node2]))
                else:
                    path = (min(node1, node2), max(node1, node2))

                    if path in self.path_list:
                        fitness += self.dist_list[self.path_list.index(path)]
                    else:
                        fitness += np.inf

            # Calculate length of final leg
            node1 = state[-1]
            node2 = state[0]

            if self.is_coords:
                fitness += np.linalg.norm(np.array(self.coords[node1])
                                          - np.array(self.coords[node2]))
            else:
                path = (min(node1, node2), max(node1, node2))

                if path in self.path_list:
                    fitness += self.dist_list[self.path_list.index(path)]
                else:
                    fitness += np.inf

            return fitness

        def get_prob_type(self):
            """ Return the problem type.

            Returns
            -------
            self.prob_type: string
                Specifies problem type as 'discrete', 'continuous', 'tsp'
                or 'either'.
            """
            return self.prob_type



    def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                    max_iters=np.inf, curve=False, random_state=None):
        """Use a standard genetic algorithm to find the optimum for a given
        optimization problem.

        Parameters
        ----------
        problem: optimization object
            Object containing fitness function optimization problem to be solved.
            For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
            :code:`TSPOpt()`.
        pop_size: int, default: 200
            Size of population to be used in genetic algorithm.
        mutation_prob: float, default: 0.1
            Probability of a mutation at each element of the state vector
            during reproduction, expressed as a value between 0 and 1.
        max_attempts: int, default: 10
            Maximum number of attempts to find a better state at each step.
        max_iters: int, default: np.inf
            Maximum number of iterations of the algorithm.
        curve: bool, default: False
            Boolean to keep fitness values for a curve.
            If :code:`False`, then no curve is stored.
            If :code:`True`, then a history of fitness values is provided as a
            third return value.
        random_state: int, default: None
            If random_state is a positive integer, random_state is the seed used
            by np.random.seed(); otherwise, the random seed is not set.

        Returns
        -------
        best_state: array
            Numpy array containing state that optimizes the fitness function.
        best_fitness: float
            Value of fitness function at best state.
        fitness_curve: array
            Numpy array of arrays containing the fitness of the entire population
            at every iteration.
            Only returned if input argument :code:`curve` is :code:`True`.

        
        """
        if pop_size < 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        if (mutation_prob < 0) or (mutation_prob > 1):
            raise Exception("""mutation_prob must be between 0 and 1.""")

        if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
           or (max_attempts < 0):
            raise Exception("""max_attempts must be a positive integer.""")

        if (not isinstance(max_iters, int) and max_iters != np.inf
                and not max_iters.is_integer()) or (max_iters < 0):
            raise Exception("""max_iters must be a positive integer.""")

        # Set random seed
        if isinstance(random_state, int) and random_state > 0:
            np.random.seed(random_state)

        if curve:
            fitness_curve = []

        # Initialize problem, population and attempts counter
        problem.reset()
        problem.random_pop(pop_size)
        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1

            # Calculate breeding probabilities
            problem.eval_mate_probs()

            # Create next generation of population
            next_gen = []

            for _ in range(pop_size):
                # Select parents
                selected = np.random.choice(pop_size, size=2,
                                            p=problem.get_mate_probs())
                parent_1 = problem.get_population()[selected[0]]
                parent_2 = problem.get_population()[selected[1]]

                # Create offspring
                child = problem.reproduce(parent_1, parent_2, mutation_prob)
                next_gen.append(child)

            next_gen = np.array(next_gen)
            problem.set_population(next_gen)

            next_state = problem.best_child()
            next_fitness = problem.eval_fitness(next_state)

            # If best child is an improvement,
            # move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

            if curve:
                fitness_curve.append(problem.get_fitness())

        best_fitness = problem.get_maximize()*problem.get_fitness()
        best_state = problem.get_state()

        if curve:
            return best_state, best_fitness, np.asarray(fitness_curve)

        return best_state, best_fitness


    def mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
              max_iters=np.inf, curve=False, random_state=None, fast_mimic=False):
        """Use MIMIC to find the optimum for a given optimization problem.

        Parameters
        ----------
        problem: optimization object
            Object containing fitness function optimization problem to be solved.
            For example, :code:`DiscreteOpt()` or :code:`TSPOpt()`.
        pop_size: int, default: 200
            Size of population to be used in algorithm.
        keep_pct: float, default: 0.2
            Proportion of samples to keep at each iteration of the algorithm,
            expressed as a value between 0 and 1.
        max_attempts: int, default: 10
            Maximum number of attempts to find a better neighbor at each step.
        max_iters: int, default: np.inf
            Maximum number of iterations of the algorithm.
        curve: bool, default: False
            Boolean to keep fitness values for a curve.
            If :code:`False`, then no curve is stored.
            If :code:`True`, then a history of fitness values is provided as a
            third return value.
        random_state: int, default: None
            If random_state is a positive integer, random_state is the seed used
            by np.random.seed(); otherwise, the random seed is not set.
        fast_mimic: bool, default: False
            Activate fast mimic mode to compute the mutual information in
            vectorized form. Faster speed but requires more memory.

        Returns
        -------
        best_state: array
            Numpy array containing state that optimizes the fitness function.
        best_fitness: float
            Value of fitness function at best state.
        fitness_curve: array
            Numpy array containing the fitness at every iteration.
            Only returned if input argument :code:`curve` is :code:`True`.

        Note
        ----
        MIMIC cannot be used for solving continuous-state optimization problems.
        """
        if problem.get_prob_type() == 'continuous':
            raise Exception("""problem type must be discrete or tsp.""")

        if pop_size < 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        if (keep_pct < 0) or (keep_pct > 1):
            raise Exception("""keep_pct must be between 0 and 1.""")

        if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
           or (max_attempts < 0):
            raise Exception("""max_attempts must be a positive integer.""")

        if (not isinstance(max_iters, int) and max_iters != np.inf
                and not max_iters.is_integer()) or (max_iters < 0):
            raise Exception("""max_iters must be a positive integer.""")

        # Set random seed
        if isinstance(random_state, int) and random_state > 0:
            np.random.seed(random_state)

        if curve:
            fitness_curve = []

        if fast_mimic not in (True, False):
            raise Exception("""fast_mimic mode must be a boolean.""")
        else:
            problem.mimic_speed = fast_mimic

        # Initialize problem, population and attempts counter
        problem.reset()
        problem.random_pop(pop_size)
        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1

            # Get top n percent of population
            problem.find_top_pct(keep_pct)

            # Update probability estimates
            problem.eval_node_probs()

            # Generate new sample
            new_sample = problem.sample_pop(pop_size)
            problem.set_population(new_sample)

            next_state = problem.best_child()

            next_fitness = problem.eval_fitness(next_state)

            # If best child is an improvement,
            # move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

            if curve:
                fitness_curve.append(problem.get_fitness())

        best_fitness = problem.get_maximize()*problem.get_fitness()
        best_state = problem.get_state().astype(int)

        if curve:
            return best_state, best_fitness, np.asarray(fitness_curve)

        return best_state, best_fitness



    # Create list of city coordinates
    
    coords_list = dp

    # Initialize fitness function object using coords_list
    fitness_coords = TravellingSales(coords = coords_list)

    # Create list of distances between pairs of cities
    

    problem_fit = opt_probs.TSPOpt(length = len(dp), fitness_fn = fitness_coords,
                                maximize=False)
    coords_list = dp
    problem_no_fit = opt_probs.TSPOpt(length = len(dp), coords = coords_list,
                                   maximize=False)
    # Solve problem using the genetic algorithm
    best_state, best_fitness =genetic_alg(problem_fit, random_state = 2)

    print('The best state found is: ', best_state)

    print('The fitness at the best state is: ', best_fitness)
    for val in best_state:
        npl.append(val)
    #print(npl)
    mysecondplot(npl,best_fitness)

def mysecondplot(alist,best_fit):
    #fig1 = plt.figure()
    fig.suptitle('TSP', fontsize=14, fontweight='bold')
    def onclick1(event):
        global i1
        global clickn
        global n
       
        i1=0
        
        while(i1<len(alist)):
            plt.plot(dpx[alist[i1]], dpy[alist[i1]],marker='D' , markersize=2,color='r')  
            ax.annotate(alist[i1], (dpx[alist[i1]], dpy[alist[i1]]),xytext=(dpx[alist[i1]]+0.15,  dpy[alist[i1]]+0.3),color='blue',size=14)
            if(i1==0):
                clickn=1
            else:
                plt.arrow(dpx[alist[i1-1]],dpy[alist[i1-1]],dpx[alist[i1]]-dpx[alist[i1-1]],dpy[alist[i1]]-dpy[alist[i1-1]],width=0.1,facecolor='yellow')
            i1+=1
            #print (i1)
        plt.text(0.25,0.25, "Minimum distance for TSP: "+str(best_fit))
        plt.ylabel('Using algo')
        plt.show()
    i1=0
    clickn=0
    n=len(alist)
    axcut = fig.add_axes([0.0, 0.0, 0.1, 0.075])
    bcut = Button(axcut, 'Sol', color='Blue', hovercolor='green')
    bcut.on_clicked(onclick1)
    ax = fig.add_subplot(212)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    plt.show()

       
        
#---------------main--------------------------

i=0
click_number=0
#cid = fig.canvas.mpl_connect('button_press_event', onclick)
ax = fig.add_subplot(211)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
cid = fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', press)
#bcut.on_clicked(_yes)
plt.show()


