import random


# %%
## Toy Search Speace

DIM = 100  # Number of bits in the bit strings (i.e. the "models").
NOISE_STDEV = 0.01  # Standard deviation of the simulated training noise.

class Model(object):
  """A class representing a model.
  
  It holds two attributes: `arch` (the simulated architecture) and `accuracy`
  (the simulated accuracy / fitness). See Appendix C for an introduction to
  this toy problem.
  
  In the real case of neural networks, `arch` would instead hold the
  architecture of the normal and reduction cells of a neural network and
  accuracy would be instead the result of training the neural net and
  evaluating it on the validation set.
  
  We do not include test accuracies here as they are not used by the algorithm
  in any way. In the case of real neural networks, the test accuracy is only
  used for the purpose of reporting / plotting final results.

  In the context of evolutionary algorithms, a model is often referred to as
  an "individual".

  Attributes:
    arch: the architecture as an int representing a bit-string of length `DIM`.
        As a result, the integers are required to be less than `2**DIM`. They
        can be visualized as strings of 0s and 1s by calling `print(model)`,
        where `model` is an instance of this class.
    accuracy:  the simulated validation accuracy. This is the sum of the
        bits in the bit-string, divided by DIM to produce a value in the
        interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
        added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
        number is clipped to within [0.0, 1.0] to produce the final validation
        accuracy of the model. A given model will have a fixed validation
        accuracy but two models that have the same architecture will generally
        have different validation accuracies due to this noise. In the context
        of evolutionary algorithms, this is often known as the "fitness".
  """
  def __init__(self):
    self.arch = None
    self.accuracy = None
    
  def __str__(self):
    """Prints a readable version of this bitstring."""
    return '{0:b}'.format(self.arch)
  
def train_and_eval(arch):
  """Simulates training and evaluation.
  
  Computes the simulated validation accuracy of the given architecture. See
  the `accuracy` attribute in `Model` class for details.
  
  Args:
    arch: the architecture as an int representing a bit-string.
  """
  accuracy =  float(_sum_bits(arch)) / float(DIM)
  accuracy += random.gauss(mu=0.0, sigma=NOISE_STDEV)
  accuracy = 0.0 if accuracy < 0.0 else accuracy
  accuracy = 1.0 if accuracy > 1.0 else accuracy
  return accuracy

def _sum_bits(arch):
  """Returns the number of 1s in the bit string.
  
  Args:
    arch: an int representing the bit string.
  """
  total = 0
  for _ in range(DIM):
    total += arch & 1
    arch = (arch >> 1)
  return total


# %%
## Search Space Traversal
def random_architecture():
  """Returns a random architecture (bit-string) represented as an int."""
  return random.randint(0, 2**DIM - 1)

def mutate_arch(parent_arch):
  """Computes the architecture for a child of the given parent architecture.

  The parent architecture is cloned and mutated to produce the child
  architecture. The child architecture is mutated by flipping a randomly chosen
  bit in its bit-string.

  Args:
    parent_arch: an int representing the architecture (bit-string) of the
        parent.

  Returns:
    An int representing the architecture (bit-string) of the child.
  """
  position = random.randint(0, DIM - 1)  # Index of the bit to flip.
  
  # Flip the bit at position `position` in `child_arch`.
  child_arch = parent_arch ^ (1 << position)
  
  return child_arch


# %%
## Regularized Evolution Algorithm
import collections

def regularized_evolution(cycles, population_size, sample_size):
  """Algorithm for regularized evolution (i.e. aging evolution).
  
  Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
  Classifier Architecture Search".
  
  Args:
    cycles: the number of cycles the algorithm should run for.
    population_size: the number of individuals to keep in the population.
    sample_size: the number of individuals that should participate in each
        tournament.

  Returns:
    history: a list of `Model` instances, representing all the models computed
        during the evolution experiment.
  """
  population = collections.deque()
  history = []  # Not used by the algorithm, only used to report results.

  # Initialize the population with random models.
  while len(population) < population_size:
    model = Model()
    model.arch = random_architecture()
    model.accuracy = train_and_eval(model.arch)
    population.append(model)
    history.append(model)

  # Carry out evolution in cycles. Each cycle produces a model and removes
  # another.
  while len(history) < cycles:
    # Sample randomly chosen models from the current population.
    sample = []
    while len(sample) < sample_size:
      # Inefficient, but written this way for clarity. In the case of neural
      # nets, the efficiency of this line is irrelevant because training neural
      # nets is the rate-determining step.
      candidate = random.choice(list(population))
      sample.append(candidate)

    # The parent is the best model in the sample.
    parent = max(sample, key=lambda i: i.accuracy)

    # Create the child model and store it.
    child = Model()
    child.arch = mutate_arch(parent.arch)
    child.accuracy = train_and_eval(child.arch)
    population.append(child)
    history.append(child)

    # Remove the oldest model.
    population.popleft()

  return history


# %%
## Example Run on the Toy Search Space
history = regularized_evolution(
    cycles=1000, population_size=100, sample_size=10)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_style('white')
xvalues = range(len(history))
yvalues = [i.accuracy for i in history]
ax = plt.gca()
ax.scatter(
    xvalues, yvalues, marker='.', facecolor=(0.0, 0.0, 0.0),
    edgecolor=(0.0, 0.0, 0.0), linewidth=1, s=1)
ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=2))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
fig = plt.gcf()
fig.set_size_inches(8, 6)
fig.tight_layout()
ax.tick_params(
    axis='x', which='both', bottom='on', top='off', labelbottom='on',
    labeltop='off', labelsize=14, pad=10)
ax.tick_params(
    axis='y', which='both', left='on', right='off', labelleft='on',
    labelright='off', labelsize=14, pad=5)
plt.xlabel('Number of Models Evaluated', labelpad=-16, fontsize=16)
plt.ylabel('Accuracy', labelpad=-30, fontsize=16)
plt.xlim(0, 1000)
sns.despine()