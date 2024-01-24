from marllib import marl

# prepare the environment
env = marl.make_env(environment_name="mpe", map_name="simple_adversary")

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

mappo.fit(env, model, stop={"timesteps_total": 21}, checkpoint_freq=20, checkpoint_end=True, share_policy="group", local_mode=True, num_workers=9)

'''mappo.render(env, model,
             restore_path={'params_path': "exp_results\mappo_mlp_simple_adversary\MAPPOTrainer_mpe_simple_adversary_caaa7_00000_0_2024-01-24_00-47-39\params.json",
                           'model_path': "exp_results\mappo_mlp_simple_adversary\MAPPOTrainer_mpe_simple_adversary_caaa7_00000_0_2024-01-24_00-47-39\checkpoint_000001\checkpoint-1", # checkpoint path
                           'render': True},  # render
             local_mode=True,
             checkpoint_end=False)'''