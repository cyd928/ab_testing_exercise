import matplotlib.pyplot as plt
import numpy as np

NUM_OF_TRAILS = 1000
EPS = 0.1  # epsilon: explored/(explored+exploited) ratio

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

def ucb(mean, n, nj):
    return mean + np.sqrt(2*(np.log(n)/nj))

# do experiment
def experiment():
    bandits = [Bandit(p) for p in bandit_prob]
    rewards = np.empty(NUM_OF_TRAILS)
    total_plays = 0

    #initialization: play each bandit once
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

    for i in range(NUM_OF_TRAILS):
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)

        # update rewards
        rewards[i] = x


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