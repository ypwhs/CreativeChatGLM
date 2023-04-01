import os
import hashlib

# 指定要检查的目录路径
directory_path = os.path.expanduser('~/.cache/huggingface/hub')

# 遍历目录树
for root, dirs, files in os.walk(directory_path):
    for dir_name in dirs:
        # 检查每个子目录是否名为"blobs"
        if dir_name == "blobs":
            # 如果是，获取该目录的绝对路径
            blobs_dir_path = os.path.join(root, dir_name)
            # 遍历blobs目录下的所有文件
            for file_name in os.listdir(blobs_dir_path):
                file_path = os.path.join(blobs_dir_path, file_name)

                # 获取文件大小，以字节为单位
                file_size = os.path.getsize(file_path)

                # 将文件大小转换为MB，并保留两位小数
                file_size_mb = round(file_size / (1024 * 1024), 2)

                # 判断文件大小是否大于100MB
                if file_size_mb < 100:
                    continue

                # 初始化哈希对象
                hash_obj = hashlib.sha256()
                # 以二进制模式打开文件
                with open(file_path, 'rb') as f:
                    # 逐块读取文件并更新哈希对象
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        hash_obj.update(chunk)
                # 获取文件的sha256哈希值
                file_hash = hash_obj.hexdigest()
                # 比较文件名和哈希值
                if file_name != file_hash:
                    print(f"Filename and hash mismatch: {file_path}, {file_hash}")
