from metaflow import (
    FlowSpec,
    step,
    current,
    Parameter,
    Config,
    secrets,
    kubernetes,
    pypi,
    card,
    gpu_profile,
    model,
    environment,
    IncludeFile,
    torchrun,
    project,
    huggingface_hub,
    # with_artifact_store,
    checkpoint,
    retry
)
from metaflow_utils import TorchTune, Accelerate
import os

k8s_config = dict(
    cpu=100,
    memory=900 * 1000,
    gpu=8,
    shared_memory=200 * 1000,
    image="registry.hub.docker.com/valayob/nebius-nccl-pytorch:0.0.2",
    # This thing needs a security context of `V1Container` with privilage=true to use Infiniband.
    disk=1000 * 1000,
    use_tmpfs=True,
)


def huggingface(func):
    deco_list = [
        pypi(
            python="3.11.5",
            packages={
                "huggingface-hub[hf_transfer]": "0.25.2"
            },  # Installing Hugging Face Hub with transfer feature
        ),
        # secrets(sources=["outerbounds.hf-wandb-keys-valay", "outerbounds.nebuis-bucket-keys"]),
        huggingface_hub(temp_dir_root="/metaflow_temp/hf_hub"),
        environment(
            vars={
                "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Enable Hugging Face transfer acceleration
            }
        ),
    ]
    for deco in deco_list:
        func = deco(func)
    return func


def training_environment(func):
    deco_list = [
        card(),
        gpu_profile(interval=10),
        pypi(
            python="3.11.10",
            packages={
                "torchtune @ git+https://github.com/pytorch/torchtune": "@8e9645c68d2e889e13607a569a419360d61760d5",
                "torch": "2.5.1",
                "torchvision": "0.20.1",
                "torchao": "0.8.0",
                "wandb": "0.19.5",
                "kagglehub": "0.3.6",  # needed by torchtune.
                "datasets": "3.2.0",
            },
        ),
        environment(
            vars={
                "WANDB_PROJECT": "grpo",
                "WANDB_LOG_MODEL": "false",
                "NCCL_IB_HCA": "mlx5",
                "UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1",
                "SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING": "1",
                "NCCL_COLLNET_ENABLE": "0",
                "OMP_NUM_THREADS": "8",
                "TORCH_DIST_INIT_BARRIER": "1"
            }
        ),
    ]
    for deco in deco_list:
        func = deco(func)
    return func


class TorchtuneGRPOSingleNodeGridSearch(FlowSpec):


    training_config = IncludeFile(
        "config",
        default="single_node_configs/3B_full_grpo_llama_32.yaml",
        is_text=True,
    )

    dry_run = Parameter("dry-run", default=False, type=bool)

    # dataset_run = Parameter("dataset-run", default="ObDocsDatasetFlow/8505", type=str)
    dataset_run = Parameter("dataset-run", default=None, type=str)

    prev_model_key = Parameter("pre-model-key", 
                            #    default='mf.models/models/artifacts/TorchTuneFlow_train_9b7c8cc5a31d41e79f63723d6dbcdec1', 
                                default=None,
                               type=str)
    
    recipe = Parameter(
        "recipe",
        default="grpo_full_finetune_distributed.py",
        help="The name of the recipe or .py file that defines the recipe. Metaflow will automatically package .py files in the flow directory."
    )

    @step
    def start(self):
        from metaflow import Run

        self.train_dataset = None
        if self.dataset_run is not None:
            final_data = Run(self.dataset_run).data.final_data
            self.train_dataset = [{"context": d} for d in final_data]

        self.next(self.pull_model)

    @huggingface
    @kubernetes(**k8s_config)
    @step
    def pull_model(self):
        # large model reference can be found here : Task("LargeModelUpload/6603/pull_model_from_huggingface/43278").data.very_large_model
        import yaml
        import time

        config = yaml.safe_load(self.training_config)
        self.model_name = config["huggingface"]["repo_id"]
        current.run.add_tag("model:%s" % self.model_name)

        start_time = time.time()
        if self.prev_model_key is None:
            self.llama_model = current.huggingface_hub.snapshot_download(
                repo_id=self.model_name,
                # force_download=True,
                allow_patterns=config["huggingface"]["allow_patterns"],
                # Download only model weights and tokenizer files
                max_workers=100,
                repo_type="model",
            )
        else: 
            self.llama_model = self.prev_model_key
        end_time = time.time()
        self.time_taken = end_time - start_time

        # TODO: Move this outside the flow / make it configurable.
        # The dynamic reward fn also needs to be improved on the torchtune side.
        self.reward_fn = [
            "default",
            "aaa_v0",
            "aaa_v1"
        ]
        self.next(
            self.train,
            foreach="reward_fn"
        )

    @retry(times=3)
    @checkpoint(
        # load_policy="eager", 
        temp_dir_root="/metaflow_temp/loaded_checkpoints"
    )
    @model(load=[("llama_model", "/metaflow_temp/llama_model")], temp_dir_root="/metaflow_temp/loaded_models")
    @training_environment
    @kubernetes(**k8s_config)
    @step
    def train(self):
        import json
        import yaml
        
        config = yaml.safe_load(self.training_config)
        config["base_model_path"] = current.model.loaded["llama_model"]
        if current.checkpoint.is_loaded:
            # If we have a checkpoint loaded because of some failure then 
            # we will also load the recipe checkpoint if it exists.
            config["base_model_path"] = current.checkpoint.directory
            if "recipe_checkpoint_key" in current.checkpoint.info.metadata:
                config["recipe_checkpoint_key"] = current.checkpoint.info.metadata["recipe_checkpoint_key"]
                recipe_checkpoint_path = current.model.load(
                    config["recipe_checkpoint_key"]
                )
                config["checkpointer"]["recipe_checkpoint"] = os.path.join(recipe_checkpoint_path, "recipe_state.pt")
                config["resume_from_checkpoint"] = True
                print("Resuming from checkpoint recipe of task:", current.checkpoint.info.pathspec, recipe_checkpoint_path)
                        
        config["run_name"] = current.pathspec
        config["output_dir"] = os.path.join(current.tempdir, "output")
        config["reward_fn"] = self.input

        if self.train_dataset is not None:
            with open("dataset_file.txt", "w") as f:
                for d in self.train_dataset:
                    f.write(d["context"] + "\n")
            config["dataset_file"] = "dataset_file.txt"

        tune = TorchTune(use_multi_node_config=False)

        tune.run(
            self.recipe,
            config_dict=config,
            additional_cli_options=["--nproc-per-node", "8"],
        )

        self.model_ref = current.model.save(
            os.path.join(
                config["output_dir"],
                # "checkpoints",
                "epoch_" + str(config["epochs"] - 1),
            ),
            storage_format="files",
        )
        self.task_id = current.task_id
        self.reward_fn_name = self.input
        self.next(self.join)

    @step
    def join(self, inputs):
        self.data = []
        for i in inputs:
            self.data.append({
                "reward_fn": i.reward_fn_name,
                "model_ref": i.model_ref,
                "task_id": i.task_id
            })
        self.next(self.end)

    @step
    def end(self):
        """End of flow"""
        print(self.data)


if __name__ == "__main__":
    TorchtuneGRPOSingleNodeGridSearch()
