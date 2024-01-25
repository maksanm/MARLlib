from marllib import marl
# import pettingzoo.mpe
# from pettingzoo.mpe import simple_adversary_v3
from pettingzoo.mpe import simple_adversary_v2
# from pettingzoo.mpe import simple_tag_v3


# prepare the environment
env = marl.make_env(environment_name="mpe", map_name="simple_adversary")

# env = simple_adversary_v2.parallel_env(render_mode="human")


# initialize algorithm and load hyperparameters
mappo = marl.algos.ippo(hyperparam_source="mpe")


# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})



# mappo.fit(env, model, stop={"timesteps_total": 100000}, checkpoint_freq=20, checkpoint_end=True, share_policy="group", local_mode=True, num_workers=9)



mappo.render(env, model,
             restore_path={'params_path': "results\ippo_mlp_mpe_simple_adversary\params.json",
                           'model_path': "results\ippo_mlp_mpe_simple_adversary\checkpoint_000089\checkpoint-89", # checkpoint path
                           'render': True},  # render
             local_mode=True,
             checkpoint_end=False)






# In[ ]:




