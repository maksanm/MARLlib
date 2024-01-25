from marllib import marl



# prepare the environment
env = marl.make_env(environment_name="mpe", map_name="simple_adversary")


# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")


# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})



mappo.render(env, model,
             restore_path={'params_path': "results\mappo_mlp_mpe_simple_adversary\params.json",
                           'model_path': "results\mappo_mlp_mpe_simple_adversary\checkpoint_000089\checkpoint-89", # checkpoint path
                           'render': True},  # render
             local_mode=True,
             checkpoint_end=False)



