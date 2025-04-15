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


class TorchtuneGRPOMultiNode(FlowSpec):

    training_config = IncludeFile(
        "config",
        default="multi_node_configs/70B_full_grpo.yaml",
        is_text=True,
    )

    recipe = Parameter(
        "recipe",
        default="grpo_full_finetune_distributed.py",
        help="The name of the recipe or .py file that defines the recipe. Metaflow will automatically package .py files in the flow directory."
    )

    dry_run = Parameter("dry-run", default=False, type=bool)

    # dataset_run = Parameter("dataset-run", default="ObDocsDatasetFlow/8931", type=str)
    dataset_run = Parameter("dataset-run", default=None, type=str)

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
        current.run.add_tag("aaa_reward")

        start_time = time.time()
        self.llama_model = current.huggingface_hub.snapshot_download(
            repo_id=self.model_name,
            # force_download=True,
            allow_patterns=config["huggingface"]["allow_patterns"],
            # Download only model weights and tokenizer files
            max_workers=100,
            repo_type="model",
        )
        end_time = time.time()
        self.time_taken = end_time - start_time
        self.next(self.train, num_parallel=4)

    @retry(times=3)
    @torchrun
    @model(load=[("llama_model", "/metaflow_temp/llama_model")])
    @training_environment
    @kubernetes(**k8s_config)
    @step
    def train(self):
        import json
        import yaml

        config = yaml.safe_load(self.training_config)

        config["run_name"] = current.pathspec
        config["output_dir"] = os.path.join(current.tempdir, "output")
        config["base_model_dir"] = current.model.loaded["llama_model"]
        # config["pbar"] = False

        if self.train_dataset is not None:
            with open("dataset_file.json", "w") as f:
                json.dump(self.train_dataset, f)
            config["dataset_file"] = "dataset_file.json"

        tune = TorchTune(use_multi_node_config=True)
        tune.run(
            self.recipe, 
            config_dict=config
        )

        if current.parallel.node_index == 0:
            self.model_ref = current.model.save(
                os.path.join(
                    config["output_dir"],
                    "checkpoints",
                    "epoch_" + str(config["epochs"] - 1),
                ),
                storage_format="files"
            )

        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the training job"""
        seen = False
        for i in inputs:
            if hasattr(i, 'model_ref'):
                if seen: 
                    print('WARNING - observed multiple model_ref in inputs of multinode step.')
                self.model_ref = i.model_ref
                seen = True
        self.next(self.end)

    @step
    def end(self):
        """End of flow"""
        pass


if __name__ == "__main__":
    TorchtuneGRPOMultiNode()
