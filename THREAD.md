# Thread draft (generated; edit freely before posting)

**1/** A model trained against a leaky eval looks brilliant in practice and fails on new data. So I built one on purpose: same RL recipe on @FireworksAI_HQ, one honest environment, one leaky. Guide and code below. 🧵 *(attach results/leak_optimizer.png or results/leak_rft.png)*

**2/** The task: a model proposes a stock-scoring formula (21 fundamentals, scored against peers). The reward is a point-in-time backtest: score each stock using only what was public that month, then check the next 12 months.

**3/** Why this testbed: the reward is verifiable and costs milliseconds, and financial data leaks in realistic ways. 10-Ks used before they were filed, restated numbers, today's index members, prices from after the decision date.

**4/** Cheap rungs first. A plain optimizer on the 21 weights, no LLM: trained honestly it earned +54.1 pts on the clean holdout. Trained on leaky data, its environment reported +32.9 pts on the training years; the truth for the same years was +22.9 pts.

**5/** Then Fireworks RFT, small open model, Eval Protocol evaluator, holdout years as the evaluation dataset for both runs. Clean run: training reward pending, clean holdout reward pending. Leaky run: pending in training, pending on the clean holdout.

**6/** One leak at a time, the most damaging was L3 price from the end of the window. The leaky formula ended up treating these price multiples as "higher is better": EV / EBITDA, Price / Book, Price / Operating cash flow, Price / Sales. The honest one: Price / Book. RL doesn't fix a leaky environment; it exploits it faster.

**7/** Friction log for the Fireworks team: [your top 2-3 from the guide].

**8/** Guide, code and numbers: [link]. If your reward is verifiable, audit the environment before you tune the algorithm. The "classroom" framing is borrowed from @chahvivi's post-training posts.
