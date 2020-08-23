import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_OF_TRAILS = 2000
bandit_prob = [0.2, 0.5, 0.75]# win rate for each bandit

# bandit class
class Bandit:
    def __init__(self, p):
        self.p = p # true win rate (unknown)
        self.a = 1
        self.b = 1
        self.N = 0. # number of samples collected so far

    def pull(self):
        # draw a 1 with prob = p
        if np.random.random() < self.p:
            res = 1
        else:
            res = 0
        return res

    def sample(self):
        return np.random.beta(self.a, self.b)

    def update(self, x):
        '''

        :param x: 0 or 1
        :return: updated win rate and number of attempts
        '''
        self.N += 1.
        self.a += x
        self.b += 1-x

def plot(bandits, trial):
    x = np.linspace(0,1,200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label = f"real p: {b.p: .4f}, win rate = {b.a-1}/{b.N}")
    plt.title(f"Bandit Distribution After {trial} Trials")
    plt.legend()
    plt.show()



def experiment():
    bandits = [Bandit(p) for p in bandit_prob]

    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    rewards = np.zeros(NUM_OF_TRAILS)

    for i in range(NUM_OF_TRAILS):
        # Thompson sampling
        j = np.argmax([b.sample() for b in bandits])

        # plot the posteriors
        if i in sample_points:
            plot(bandits, i)

        # pull the arm of the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards
        rewards[i] = x

        # update the distribution of the bandit whose arm we just pulled
        bandits[j].update(x)


# plot
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_OF_TRAILS)+1)
    plt.plot(win_rates, label='win rate')
    plt.plot(np.ones(NUM_OF_TRAILS)*np.max(bandit_prob), label='max ctr')
    plt.legend()
    plt.xscale('log')
    plt.ylim([0,1])
    plt.title('Win Rate Trend')
    plt.xlabel('Number of Trials')
    plt.ylabel('Win Rate')
    plt.show()


if __name__ =="__main__":
    experiment()