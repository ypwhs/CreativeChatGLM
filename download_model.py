from huggingface_hub import snapshot_download

snapshot_download(
	repo_id='THUDM/chatglm-6b-int4', 
	local_dir='THUDM/chatglm-6b-int4', 
	local_dir_use_symlinks=False
)
