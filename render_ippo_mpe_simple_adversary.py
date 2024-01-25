from marllib import marl

# prepare the environment
env = marl.make_env(environment_name="mpe", map_name="simple_adversary")

# initialize algorithm and load hyperparameters
algo = marl.algos.ippo(hyperparam_source="mpe_simple_adversary")

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "128-256"})

# uncomment to train new model
# algo.fit(env, model, stop={"timesteps_total": 100000}, checkpoint_freq=20, checkpoint_end=True, share_policy="group", local_mode=True, num_workers=9)

# render from checkpoint model located in model_path folder
algo.render(env, model,
             restore_path={'params_path': "results\ippo_mlp_mpe_simple_adversary\params.json",
                           'model_path': "results\ippo_mlp_mpe_simple_adversary\checkpoint_000089\checkpoint-89", # checkpoint path
                           'render': True},  # render
             local_mode=True,
             checkpoint_end=False)
