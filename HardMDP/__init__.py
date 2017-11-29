from gym.envs.registration import register

Ndict = range(1, 100)

for N in Ndict:
    register(
        id=str(N) + 'NDeterministicChain-v0',
        entry_point='HardMDP.Chain:NDeterministicChain',
        kwargs={'N': N})
