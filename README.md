The Multiarmed-bandit problem
The multi-armed bandit (MAB) is a classic problem in decision sciences. Effectively, it is one of optimal resource allocation under uncertainty. The name is derived from old slot machines that where operated by pulling an arm — they are called bandits because they rob those who play them. Now, imagine there are multiple machines and we suspect that the payout rate — the payout to pull ratio — varies across the machines. Naturally we want to identify the machine with the highest payout and exploit it — i.e. pull it more than the others.

The MAB problem is this; how do you most efficiently identify the best machine to play, whilst sufficiently exploring the many options in real time? This problem is not an exercise in theoretical abstraction, it is an analogy for a common problem that organisations face all the time, that is, how to identify the best message to present to customers (message is broadly defined here i.e. webpages, advertising, images) such that it maximises some business objective (e.g. clickthrough rate, signups).

The classic approach to making decisions across variants with unknown performance outcomes is to perform multiple A/B tests. These are typically run by evenly directing a percentage of traffic across each of the variants over a number of weeks, then performing statistical tests to identify which variant is the best. This is perfectly fine when there are a small number of variations of the message (e.g. 2–4), but can be quite inefficient in terms of both time and opportunity cost when there are many.

The time argument is easy to grasp. Simply, more variations requires more A/B tests, which take more time, thus delaying “feedback” and decision making. The opportunity cost argument is more subtle. In economics, the opportunity cost is the cost that is associated with taking one action rather than another. Simply, what did I miss out on by putting my money into investment A rather than that investment B? Investment B is the opportunity cost of taking investment A. In the variant testing world this translates to sending a customer to A rather than B.

For good reasons A/B tests should not be “peeked” at whilst the tests are running. This means that the experimenters will not know which variant is best until they end the test. However, it is typically hypothesised that one variant will outperform the others. What does this mean in real terms? It means that A/B testing involves consciously sending a proportion of your customers to a suboptimal message (though you don’t know which one!) — perhaps if these customers where sent to the optimal variant they might have signed up for your service.

This is the opportunity cost in A/B testing. For one test this is acceptable. However, when there are many variants to test, this means that you are potentially directing many customers to suboptimal variants for a long period of time. It would be better in this scenario if we could, in real time, quickly rule out the dud variants, without directing too much traffic to them; then after a number of effective variants have been identified perform an A/B test on this smaller subset (this is usually required for statistical power). Bandit algorithms or samplers, are a means of testing and optimising variant allocation quickly.

In this post I’ll provide an introduction to Thompson sampling (TS) and its properties. I’ll also compare Thompson sampling against the epsilon-greedy algorithm, which is another popular choice for MAB problems. Everything will be implemented from scratch in Python

I use the following vocabulary in the post:

Trial: A customer arriving on a webpage

Message: the image/words/colours etc. being tested

Variant: different variations of a message (ad/image/webpage etc.)

Action: The action taken by an algorithm, that is, the variant it decides to show

Reward: The business objective, for example a signup to a service and click-through. For simplicity we will assume that rewards are binomially distributed. That is, a reward is either a 1 or 0 (click through or not)

Agent: The algorithm that makes decisions concerning which variant to show. I also refer to these as bandits and samplers

Environment: The context in which the agent operates — i.e. the variants and their latent “payouts”

Exploration and Exploitation
Bandit algorithms are approaches to realtime/online decision making that strive to strike a balance between sufficiently exploring the variant space and exploiting the optimal action.

Striking a balance between the two is critically important. Firstly, the variant space needs to be sufficiently explored such that the strongest variant is identified. By first identifying then continuing to exploit the optimal action you are maximising the total reward that is available to you from the environment. However, you also want to continue to explore other feasible variants in case they provide better returns in the future. That is, you want to hedge your bets somewhat by continuing to experiment (a little) with sub-optimal variants in the event that their payouts change. If they do, your algorithm will pick up on the change and will begin selecting this variant for new customers. A further benefit of exploration is that you learn more about the generating process underlying the variant. That is, what is its average payout rate and what is distribution of uncertainty around that. The key, therefore, is to decide how best to balance this exploration-exploitation tradeoff.

Epsilon-Greedy
A common approach to balancing the exploitation-exploration tradeoff is the epilson- or e-greedy algorithm. Greedy here means what you probably think it does. After an initial period of exploration (for example 1000 trials), the algorithm greedily exploits the best option k, e percent of the time. For example, if we set e=0.05, the algorithm will exploit the best variant 95% of the time and will explore random alternatives 5% of the time. This is actually quite effective in practice, but as we’ll come to see it can under explore the variant space before exploiting what it estimates to be the strongest variant. This means that e-greedy can get stuck exploiting a suboptimal variant. Let’s dive into some code to show e-greedy in a action.

First some dependencies and boiler plate. We need to define the environment. The environment is the context in which the algorithms will run. In this case, it is very simple. The environment calls on an an agent (i.e. a bandit algorithm) to “decide” what action to take, it then runs the action, observes the reward and communicates the reward back to the agent, which updates itself. This has many simularities to reinforceent learning and, indeed, MAB algorithms are considered to be a “lite” form of RL.

The reward is binomially distributed with probability p being determined by the action taken (note: this can easily be extended to continuous outcomes). Here I also define a BaseSampler class. The purpose of this class is really only to store the various attributes and logs (for visualization) that are common across bandits. Here we also define our variants and their latent, but unknown, payout rates. In total, we will test 10 variants. The best option is variant 9, which has a payout rate of 0.11%.

variants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
payouts = [0.023, 0.03, 0.029, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.11]
For baseline comparison, I also define the RandomSampler class. This simply selects a variant at random for each trial, it does not learn, it does not update. This is purely to benchmark the other agents.


class RandomSampler(BaseSampler):
    def __init__(self, env):
        super().__init__(env)
        
    def choose_k(self):
        
        self.k = np.random.choice(self.variants)
        
        return self.k
    
    def update(self):
        # nothing to update
        #self.thetaregret[self.i] = self.thetaregret[self.i]
        #self.regret_i[self.i] = np.max(self.thetaregret) - self.theta[self.k]
        self.thetaregret[self.i] = np.max(self.theta) - self.theta[self.k]
        
        self.a[self.k] += self.reward
        self.b[self.k] += 1
        self.theta = self.a/self.b

        self.ad_i[self.i] = self.k
        self.r_i[self.i] = self.reward
        self.i += 1
The other agents follow this basic structure. They all implement choose_k and update methods. choose_k implements the policy through which the agent selects a variant. update updates the parameters of the agent — this is how the agent “evolves” its ability to select a variant (the RandomSampler class doesn’t update anything). We run an agent in an environment using this pattern:

en0 = Environment(machines, payouts, n_trials=10000)
rs = RandomSampler(env=en0)
en0.run(agent=rs)
Descriptions of each component of the e-greedy algorithm are inline, but the core of the algorithm is this:

- randomly choose k for n trials
- On each trial estimate the payout rate for each variant
- after n learning trials:
- select 1-e% of the time k with the the highest payout rate and;
- e% of the time sample from the variants randomly

class eGreedy(BaseSampler):

    def __init__(self, env, n_learning, e):
        super().__init__(env, n_learning, e)
        
    def choose_k(self):

        # e% of the time take a random draw from machines
        # random k for n learning trials, then the machine with highest theta
        self.k = np.random.choice(self.variants) if self.i < self.n_learning else np.argmax(self.theta)
        # with 1 - e probability take a random sample (explore) otherwise exploit
        self.k = np.random.choice(self.variants) if self.ep[self.i] > self.exploit else self.k
        return self.k

    def update(self):
        
        # update the probability of payout for each machine
        self.a[self.k] += self.reward
        self.b[self.k] += 1
        self.theta = self.a/self.b

        self.thetas[self.i] = self.theta[self.k]
        self.thetaregret[self.i] = np.max(self.thetas) - self.theta[self.k]
 
        self.ad_i[self.i] = self.k
        self.r_i[self.i] = self.reward
        self.i += 1
        
        https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50
