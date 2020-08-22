import matplotlib.pyplot as plt
import numpy as np

NUM_OF_TRAILS = 1000
EPS = 0.1  # epsilon
bandit_prob = [0.2, 0.5, 0.75]# win rate for each bandit

# bandit class
class Bandit:
    def __init__(self, p):
        self.p = p # true win rate (unknown)
        self.p_estimate = 0. # est. p
        self.N = 0. # number of samples collected so far

    def pull(self):
        # draw a 1 with prob = p
        if np.random.random() < self.p:
            res = 1
        else:
            res = 0
        return res

    def update(self, x):
        '''

        :param x: 0 or 1
        :return: updated win rate and number of attempts
        '''
        self.N += 1.
        self.p_estimate += ((self.N-1)*self.p_estimate + x)/self.N

# epsilon-greedy loop
def experiment():
    bandits = [Bandit(p) for p in bandit_prob]

    rewards = np.zeros(NUM_OF_TRAILS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal= 0

    optimal_j = np.argmax([b.p for b in bandits]) #index for the bandit with max true mean (unknown in reality)
    print('optimal j:', optimal_j)

    for i in range(NUM_OF_TRAILS):
        # use epsilon-greedy to select the next bandit
        if np.random.random() < EPS:
            num_times_explored += 1
            j = np.random.randint(len(bandits)) # index of a random bandit
        else:
            num_times_exploited += 1
            j = np.argmax([b.p for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        # pull the arm of the bandit with the largest sample
        x = bandits[j].pull()

        # update rewards
        rewards[i] = x

        # update the distribution of the bandit whose arm we just pulled
        bandits[j].update(x)


# print results & win rate
    for b in bandits:
        print('mean estimate:', b.p_estimate)
        print('total reward earned:', rewards.sum())
        print('overall win rate:', rewards.sum()/NUM_OF_TRAILS)
        print('number of times explored / exploited:', num_times_explored,'/', num_times_exploited)
# plot
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_OF_TRAILS)+1)
    plt.plot(win_rates, label='win rate')
    plt.plot(np.ones(NUM_OF_TRAILS)*np.max(bandit_prob), label='max ctr')
    plt.legend()
    plt.title('Win Rate Trend')
    plt.xlabel('Number of Trials')
    plt.ylabel('Win Rate')
    plt.show()


if __name__ =="__main__":
    experiment()