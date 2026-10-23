import wandb

def log_artifact(artifact_name, artifact_type, artifact_description, filename, wandb_run):
    """
    Log the provided filename as an artifact in W&B, and add the artifact path to the MLFlow run
    so it can be retrieved by subsequent steps in a pipeline
    """
    artifact = wandb.Artifact(
        artifact_name,
        type=artifact_type,
        description=artifact_description,
    )
    artifact.add_file(filename)
    wandb_run.log_artifact(artifact)
