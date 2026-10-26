import logging
import os

log = logging.getLogger(__name__)


class HubUploader(object):
    """
    Pushes checkpoints and the resolved Hydra config of a run to a Hugging Face model repository.

    Authentication: HF_TOKEN environment variable or a prior `huggingface-cli login`.
    Upload errors are logged and never stop the training.
    """

    def __init__(self, repo_id, private=True, path_in_repo=None):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo
        self.api.create_repo(repo_id, private=private, exist_ok=True)
        log.info(f"Checkpoints go to https://huggingface.co/{repo_id}/tree/main/{path_in_repo or ''}")

    def upload(self, local_dir, message, patterns=('*.pth', '.hydra/*.yaml', '*.png')):
        try:
            self.api.upload_folder(repo_id=self.repo_id, folder_path=local_dir, path_in_repo=self.path_in_repo,
                                   allow_patterns=list(patterns), commit_message=message)
        except Exception as e:  # network, quota, auth
            log.warning(f"Upload to {self.repo_id} failed: {e}")

    @staticmethod
    def from_config(repo_id, private, run_name):
        if not repo_id:
            return None
        if not os.environ.get('HF_TOKEN'):
            log.info('HF_TOKEN not set, using the token of huggingface-cli login if there is one')
        return HubUploader(repo_id, private=private, path_in_repo=f"runs/{run_name}")
