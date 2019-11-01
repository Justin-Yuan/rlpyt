notes to help decompose and modify the rlpyt repo 

--- 

## How to run 

build_and_train
- args: game (name of task/env), run_ID, cuda_idx
- make sampler 
- make algo (e.g. DQN)
- make agent 
- make runner 
- make config 
- runner.train()

--- 

## rlpyt/algos

### base.py 

RLAlgorithm 
- args: agent, n_itr, batch_spec, mid_batch_reset, examples, world_size, rank 

### pg/base.py

PolicyGradientAlgo 
- process_returns
    - inputs: samples
    - outputs: return_, advantage, valid 

```python
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "entropy", "perplexity"])
AgentTrain = namedtuple("AgentTrain", ["dist_info", "value"])
```

### pg/ppo.py 

PPO
- optimize_agent 
    - inputs: itr, samples
    - outputs: opt_info 
    - construct agent inputs
        - experience tuples (obs, prev_action, prev_reward)
    - construct loss inputs
    - for in range(epochs)
        - for in range(mini_batches)
            - derive loss 
            - optimzer step  
- loss 
    - inputs: agent_inputs, action, return_, advantage, valid, old_dist_info, init_rnn_state
    - outputs: loss, entropy, perplexity  
    - run agent model forward 
    ```python
    dist_info, value = self.agent(*agent_inputs)
    dist = self.agent.distribution
    ```
    - calculate ppo loss, value loss (TD loss) and entropy 


--- 

## rlpyt/samplers 

### base.py 

BaseSampler 
- interface with runner, in master process only 
- args: EnvCls, ensv_kwargs, batch_T, batch_B, CollectorCls 
- obtain_samples 
    - inputs: itr 
- evaluate_agents
    - inputs: itr 

### collector.py 

BaseCollector
- steps through environments, in worker process 
- args: rank, envs, samples_np, batch_T, TrajInfoCls 
- start_envs 
- start_agent 
- collect_batch 
    - inputs: agent_inputs, traj_infos 

### serial/sampler.py 

SerialSampler 


--- 

## rlpyt/envs

### base.py 

Env
- property: observation_space, action_space, spaces, horizon 
- step 
    - inputs: action 
    - outputs: (obs, reward, done, info)
- reset 
- close 


--- 

## rlpyt/runners 

### base.py 

BaseRunner 
- train 


---

## rlpyt/models


--- 

## rlpyt/utils 

namedarraytuple 