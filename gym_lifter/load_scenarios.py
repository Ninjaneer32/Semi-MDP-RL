from gym_lifter.envs.fab.gen_scenarios import generate_scenarios

for pod in [False, True]:
    for f in [8, 10, 12, 16, 20]:
        generate_scenarios(f, seed=20230217, pod=pod)
