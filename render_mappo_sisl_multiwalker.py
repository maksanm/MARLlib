from marllib import marl
import warnings
warnings.filterwarnings("ignore")

# use settings from marllib/envs/base_env/config/sisl.yaml
env = marl.make_env(environment_name="sisl", map_name="multiwalker")

# initialize algorithm with finetuned hyper-parameters located in marllib/marl/algos/hyperparams/finetuned/sisl
algo = marl.algos.mappo(hyperparam_source="sisl_multiwalker")

# build agent model based on env ironment + algorithm + user preference
model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "128-256"})

# uncomment to train new model
# algo.fit(env, model, stop={"timesteps_total": 10000000}, checkpoint_freq=20, share_policy="group", local_mode=True, num_workers=9)

# render from checkpoint model located in model_path folder
algo.render(env, model,
             restore_path={'params_path': "results/mappo_mlp_multiwalker/params.json",
                           'model_path': f"results/mappo_mlp_multiwalker/checkpoint_000180/checkpoint-180",
                           'render': True},
             local_mode=True,
             share_policy="group",
             checkpoint_end=False)
