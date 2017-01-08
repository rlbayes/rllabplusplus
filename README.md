# rllab++

rllab++ is a framework for developing and evaluating reinforcement learning algorithms, built on [rllab](https://github.com/openai/rllab). It has the following implementations besides the ones implemented in rllab:

- [Q-Prop](https://arxiv.org/abs/1611.02247)

# Installation

Please follow the basic installation instructions in [rllab documentation](https://rllab.readthedocs.io/en/latest/), with the following minor changes:

- Install tensorflow-0.11.0rc0

# Examples

From the [launchers](/sandbox/rocky/tf/launchers) directory, run the following, with optional additional flags defined in [launcher_utils.py](/sandbox/rocky/tf/launchers/launcher_utils.py):

```
python algo_gym_stub.py --exp=<exp_name> 
```

Flags include:

- algo\_name: trpo ([TRPO](https://arxiv.org/abs/1502.05477)), vpg (vanilla policy gradient), ddpg ([DDPG](https://arxiv.org/abs/1603.00748)), qprop ([Q-Prop](https://arxiv.org/abs/1611.02247) with trpo), qvpg (Q-Prop with vpg).
- env\_name: [OpenAI Gym](https://gym.openai.com/) environment name, e.g. HalfCheetah-v1.

The experiment will be saved in /data/local/\<exp\_name\>. To view the results, run [viskit](/rllab/viskit/frontend.py) or [viskit2](/rllab/viskit/frontend2.py) with the path to the experiment folders as arguments.

# Citations

If you use rllab++ for academic research, you are highly encouraged to cite the following papers:

- Shixiang Gu, Timothy Lillicrap, Zoubin Ghahramani, Richard E. Turner, Sergey Levine. "[Q-Prop: Sample-Efficient Policy Gradient with an Off-Policy Critic](https://arxiv.org/abs/1611.02247)". _arXiv:1611.02247 [cs.LG], 2016._
- Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. "[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778)". _Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016._

