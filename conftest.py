import wandb

def pytest_sessionstart(session):
    # Automatically initialize a W&B run when pytest starts
    wandb.init(job_type="data_check")

def pytest_sessionfinish(session, exitstatus):
    # Safely close the W&B run when pytest finishes
    wandb.finish()