"""
用于下载和可视化Hugging Face数据集的脚本，特别是opencsg/chinese-fineweb-edu-v2数据集
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, config
from tqdm import tqdm
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from collections import Counter
import jieba
import hashlib
import json
import shutil
import glob
import re
import requests
from huggingface_hub import hf_hub_download, scan_cache_dir, HfApi
from huggingface_hub.utils import validate_repo_id
from datasets.utils.file_utils import get_datasets_user_agent
from pathlib import Path
import tempfile
import time

# 设置Hugging Face的缓存目录
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\huggingface_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\huggingface_cache\datasets"

# 使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="下载和可视化Hugging Face数据集")
    parser.add_argument("--dataset", type=str, default="opencsg/chinese-fineweb-edu-v2", 
                       help="要下载的数据集名称")
    parser.add_argument("--split", type=str, default="train", 
                       help="要下载的数据集分割")
    parser.add_argument("--streaming", action="store_true", 
                       help="是否流式加载数据集")
    parser.add_argument("--sample_size", type=int, default=1000, 
                       help="要分析的样本数量")
    parser.add_argument("--save_dir", type=str, default="dataset_visualization", 
                       help="保存可视化结果的目录")
    parser.add_argument("--text_column", type=str, default="text", 
                       help="数据集中文本列的名称")
    parser.add_argument("--cache_dir", type=str, default=None, 
                       help="缓存数据集的目录")
    parser.add_argument("--download_only", action="store_true", 
                       help="仅下载数据集，不进行可视化")
    parser.add_argument("--force_download", action="store_true",
                       help="强制重新下载数据集，忽略缓存")
    parser.add_argument("--verify_downloads", action="store_true",
                       help="验证缓存文件的完整性")
    return parser.parse_args()


def show_cache_paths():
    """显示当前Hugging Face缓存路径"""
    print("\n当前Hugging Face缓存路径:")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
    print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', '未设置')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', '未设置')}")
    print()


def get_dataset_metadata(dataset_name: str) -> Dict:
    """获取数据集的元数据，包括可能的MD5校验和信息
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        Dict: 数据集的元数据
    """
    try:
        # 尝试获取Croissant元数据，其中可能包含MD5信息
        url = f"https://huggingface.co/api/datasets/{dataset_name}/croissant"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # 尝试获取普通的数据集信息
            api = HfApi()
            info = api.dataset_info(dataset_name)
            return info.__dict__
    except Exception as e:
        print(f"获取数据集元数据时出错: {e}")
        return {}


def get_parquet_files_info(dataset_name: str) -> List[Dict]:
    """获取数据集的parquet文件列表和相关信息
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        List[Dict]: parquet文件信息列表
    """
    try:
        # 获取parquet文件列表
        url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("parquet_files", [])
        else:
            return []
    except Exception as e:
        print(f"获取parquet文件列表时出错: {e}")
        return []


def get_file_md5_checksum(file_path: str) -> str:
    """计算文件的MD5校验和
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: MD5校验和
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_md5_from_metadata(metadata: Dict, file_name: str) -> Optional[str]:
    """从元数据中提取特定文件的MD5校验和
    
    Args:
        metadata: 数据集元数据
        file_name: 文件名
        
    Returns:
        Optional[str]: MD5校验和，如果未找到则返回None
    """
    # 检查Croissant格式的元数据，其中可能包含MD5
    if "@context" in metadata and "md5" in metadata.get("@context", {}):
        # 在distribution部分中寻找文件
        for item in metadata.get("distribution", []):
            if item.get("name") and file_name in item.get("name", ""):
                return item.get("md5")
            if item.get("includes") and file_name.endswith(item.get("includes", "").split("/")[-1]):
                return item.get("md5")
    
    # 检查dataset_info.json格式
    if "dataset_info" in metadata:
        dataset_info = metadata["dataset_info"]
        if "download_checksums" in dataset_info:
            for file_path, info in dataset_info.get("download_checksums", {}).items():
                if file_name in file_path:
                    return info.get("md5", None)
    
    return None


def get_parquet_file_info(dataset_name: str, cache_dir: Optional[str] = None) -> List[Dict]:
    """获取数据集的parquet文件信息
    
    Args:
        dataset_name: 数据集名称
        cache_dir: 缓存目录
        
    Returns:
        List[Dict]: 文件信息列表，包含文件路径和预期大小等信息
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
    
    # 扫描缓存目录，查找与数据集相关的所有parquet文件
    parquet_files = []
    
    try:
        # 获取在线的parquet文件信息
        online_parquet_files = get_parquet_files_info(dataset_name)
        
        # 获取数据集元数据，可能包含MD5信息
        metadata = get_dataset_metadata(dataset_name)
        
        # 扫描缓存目录找到可能的数据集信息文件
        cache_info = scan_cache_dir(cache_dir)
        dataset_repos = [repo for repo in cache_info.repos if dataset_name in repo.repo_id]
        
        if not dataset_repos:
            print(f"未找到数据集 {dataset_name} 的缓存信息")
            return []
        
        # 查找所有parquet文件目录
        for repo in dataset_repos:
            for revision in repo.revisions:
                # 尝试找到数据集信息文件
                info_files = [f for f in revision.snapshot_files if f.file_path.endswith('dataset_info.json')]
                
                if info_files:
                    # 获取parquet文件所在的目录
                    for info_file in info_files:
                        info_file_path = os.path.join(cache_dir, info_file.file_path)
                        info_dir = os.path.dirname(info_file_path)
                        
                        # 尝试从info文件中获取有关parquet文件的信息
                        try:
                            with open(info_file_path, 'r', encoding='utf-8') as f:
                                dataset_info = json.load(f)
                                
                            # 查找同目录下的所有parquet文件
                            pattern = os.path.join(info_dir, '*.parquet')
                            found_files = glob.glob(pattern)
                            
                            # 提取文件编号用于排序
                            def extract_number(filepath):
                                match = re.search(r'(\d+)\.parquet$', filepath)
                                return int(match.group(1)) if match else 0
                            
                            # 按编号排序文件
                            found_files.sort(key=extract_number)
                            
                            # 获取每个文件的信息
                            for file_path in found_files:
                                file_name = os.path.basename(file_path)
                                
                                # 查找对应的在线文件信息
                                online_info = next((f for f in online_parquet_files if f.get("filename") == file_name), {})
                                
                                # 从元数据中提取MD5信息
                                expected_md5 = extract_md5_from_metadata(metadata, file_name)
                                if not expected_md5 and "dataset_info" in dataset_info:
                                    expected_md5 = extract_md5_from_metadata({"dataset_info": dataset_info}, file_name)
                                
                                file_info = {
                                    'file_path': file_path,
                                    'file_name': file_name,
                                    'expected_size': online_info.get("size"),
                                    'expected_md5': expected_md5,
                                    'is_downloaded': os.path.exists(file_path)
                                }
                                
                                # 如果文件已下载，获取当前大小
                                if file_info['is_downloaded']:
                                    file_info['current_size'] = os.path.getsize(file_path)
                                else:
                                    file_info['current_size'] = 0
                                
                                parquet_files.append(file_info)
                        except Exception as e:
                            print(f"解析数据集信息文件失败: {e}")
        
        return parquet_files
    
    except Exception as e:
        print(f"获取parquet文件信息时出错: {e}")
        return []


def verify_parquet_file(file_path: str, expected_md5: Optional[str] = None) -> bool:
    """验证parquet文件的完整性
    
    Args:
        file_path: parquet文件路径
        expected_md5: 预期的MD5校验和，如果提供则会进行校验
        
    Returns:
        bool: 文件是否完整
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    try:
        # 如果提供了MD5校验和，先验证MD5
        if expected_md5:
            print(f"开始计算文件MD5校验和...")
            actual_md5 = get_file_md5_checksum(file_path)
            print(f"文件MD5: {actual_md5}")
            if actual_md5 != expected_md5:
                print(f"MD5校验失败: 预期 {expected_md5}, 实际 {actual_md5}")
                return False
            print("MD5校验通过")
        
        # 尝试读取parquet文件
        print(f"尝试读取parquet文件内容...")
        df = pd.read_parquet(file_path)
        rows = len(df)
        cols = len(df.columns)
        print(f"成功读取parquet文件，包含 {rows} 行，{cols} 列")
        # 如果能成功读取，说明文件完整
        return True
    except Exception as e:
        print(f"验证parquet文件 {file_path} 失败: {e}")
        return False


def verify_cache_files(dataset_name: str, cache_dir: Optional[str] = None) -> Tuple[bool, List[Dict]]:
    """验证数据集缓存文件的完整性
    
    Args:
        dataset_name: 数据集名称
        cache_dir: 缓存目录
        
    Returns:
        Tuple[bool, List[Dict]]: (是否所有文件都完整, 不完整文件列表)
    """
    print(f"正在验证数据集 {dataset_name} 的缓存文件完整性...")
    
    if cache_dir is None:
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
    
    try:
        # 获取所有parquet文件信息
        parquet_files = get_parquet_file_info(dataset_name, cache_dir)
        
        if not parquet_files:
            print(f"未找到数据集 {dataset_name} 的parquet文件")
            return False, []
        
        # 验证每个parquet文件
        incomplete_files = []
        all_complete = True
        
        for file_info in parquet_files:
            file_path = file_info['file_path']
            file_name = file_info['file_name']
            expected_md5 = file_info.get('expected_md5')
            
            print(f"验证文件: {file_name}...")
            
            if not file_info['is_downloaded'] or not verify_parquet_file(file_path, expected_md5):
                print(f"文件不完整或损坏: {file_name}")
                all_complete = False
                incomplete_files.append(file_info)
            else:
                print(f"文件验证通过: {file_name}")
        
        if all_complete:
            print("所有parquet文件验证通过")
        else:
            print(f"发现 {len(incomplete_files)} 个不完整的文件")
        
        return all_complete, incomplete_files
    
    except Exception as e:
        print(f"验证缓存文件时出错: {e}")
        return False, []


def clean_dataset_cache(dataset_name: str, files_to_clean: List[Dict] = None, cache_dir: Optional[str] = None):
    """清理数据集缓存
    
    Args:
        dataset_name: 数据集名称
        files_to_clean: 需要清理的文件列表，如果为None则清理所有相关文件
        cache_dir: 缓存目录
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
    
    try:
        if files_to_clean:
            # 只清理指定的文件
            for file_info in files_to_clean:
                file_path = file_info['file_path']
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"已删除不完整的缓存文件: {file_info['file_name']}")
        else:
            # 扫描缓存目录，清理所有相关文件
            cache_info = scan_cache_dir(cache_dir)
            
            # 查找与数据集相关的缓存文件并删除
            for repo in cache_info.repos:
                if dataset_name in repo.repo_id:
                    for revision in repo.revisions:
                        for cache_file in revision.snapshot_files:
                            file_path = os.path.join(cache_dir, cache_file.file_path)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"已删除缓存文件: {file_path}")
                        
                        # 尝试删除空目录
                        if revision.snapshot_files:
                            revision_dir = os.path.dirname(os.path.join(cache_dir, revision.snapshot_files[0].file_path))
                            if os.path.exists(revision_dir) and not os.listdir(revision_dir):
                                os.rmdir(revision_dir)
            
            print(f"已清理数据集 {dataset_name} 的所有缓存")
    
    except Exception as e:
        print(f"清理缓存文件时出错: {e}")


def download_dataset(dataset_name: str, split: str, streaming: bool = False, 
                    cache_dir: Optional[str] = None, force_download: bool = False,
                    verify_downloads: bool = False) -> Any:
    """下载指定的Hugging Face数据集
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        streaming: 是否流式加载
        cache_dir: 缓存目录
        force_download: 是否强制重新下载
        verify_downloads: 是否验证下载文件完整性
        
    Returns:
        加载的数据集对象
    """
    print(f"正在下载数据集: {dataset_name}, 分割: {split}")
    
    # 如果未指定cache_dir，使用环境变量设置的路径
    if cache_dir is None:
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
        print(f"使用默认缓存目录: {cache_dir}")
    else:
        print(f"使用自定义缓存目录: {cache_dir}")
    
    # 自定义数据集下载类，用于实现顺序下载和详细日志
    class SequentialDownloader:
        def __init__(self, dataset_name, split, cache_dir, verify_downloads=True):
            self.dataset_name = dataset_name
            self.split = split
            self.cache_dir = cache_dir
            self.verify_downloads = verify_downloads
            
            # 获取数据集名称的各种变体
            self.dataset_org = dataset_name.split('/')[0] if '/' in dataset_name else ''
            self.dataset_name_only = dataset_name.split('/')[-1]
            
            # 预定义可能的下载目录格式
            self.dir_formats = [
                # 标准格式
                os.path.join(cache_dir, dataset_name.replace("/", "---"), split),
                # 替代格式1
                os.path.join(cache_dir, dataset_name.replace("/", "--"), split),
                # 替代格式2
                os.path.join(cache_dir, f"{self.dataset_org}--{self.dataset_name_only}" if self.dataset_org else self.dataset_name_only, split)
            ]
            
            self.download_mode = "force_redownload" if force_download else "reuse_dataset_if_exists"
        
        def find_download_dir(self):
            """找到下载目录或创建一个新目录"""
            print("\n查找可能的下载目录:")
            for dir_path in self.dir_formats:
                if os.path.exists(dir_path):
                    print(f"找到已存在的下载目录: {dir_path}")
                    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.parquet')])
                    if files:
                        print(f"目录中已有 {len(files)} 个parquet文件: {files}")
                        return dir_path
                    else:
                        print(f"目录存在但为空")
                else:
                    print(f"目录不存在: {dir_path}")
            
            print(f"未找到有效的下载目录，将在首次下载时自动创建")
            return None
        
        def check_file_exists(self, file_index):
            """检查指定索引的文件是否存在并完整
            
            Args:
                file_index: 文件索引(0, 1, 2, ...)
            
            Returns:
                (exists, file_path): 文件是否存在和完整，以及文件路径
            """
            file_name = f"{file_index:04d}.parquet"  # 格式化为0000.parquet, 0001.parquet等
            
            print(f"\n检查文件 {file_name} 是否存在:")
            
            # 先检查标准目录格式
            for dir_path in self.dir_formats:
                file_path = os.path.join(dir_path, file_name)
                print(f"检查路径: {file_path}")
                
                if os.path.exists(file_path):
                    print(f"文件存在: {file_path}")
                    file_size = os.path.getsize(file_path)
                    print(f"文件大小: {file_size / (1024*1024):.2f} MB")
                    
                    if file_size == 0:
                        print(f"文件大小为0，判定为不完整")
                        return False, file_path
                    
                    # 如果需要验证，则验证文件完整性
                    if self.verify_downloads:
                        print(f"验证文件完整性...")
                        try:
                            is_valid = verify_parquet_file(file_path)
                            if is_valid:
                                print(f"文件验证通过，无需重新下载: {file_name}")
                                return True, file_path
                            else:
                                print(f"文件验证失败，需要重新下载: {file_name}")
                                # 删除不完整文件
                                try:
                                    os.remove(file_path)
                                    print(f"已删除不完整文件: {file_path}")
                                except Exception as e:
                                    print(f"删除文件失败: {e}")
                                return False, file_path
                        except Exception as e:
                            print(f"验证文件时出错: {e}")
                            return False, file_path
                    else:
                        # 不验证的情况下，只要文件存在且大小不为0就认为完整
                        print(f"跳过验证，文件存在且大小不为0，无需重新下载: {file_name}")
                        return True, file_path
            
            # 如果在标准目录中找不到，尝试在数据集缓存中进行更广泛的搜索
            print(f"在标准目录中未找到文件，尝试进行更广泛的搜索...")
            
            # 获取数据集名称的各种变体用于搜索
            dataset_org = self.dataset_name.split('/')[0] if '/' in self.dataset_name else ''
            dataset_name_only = self.dataset_name.split('/')[-1]
            
            # 构建可能的缓存路径模式
            possible_paths = [
                # 开始检查是否有其他格式的文件名可以重命名
                os.path.join(self.cache_dir, "**", f"*{file_index}*.parquet"),  # 任何包含该索引的parquet文件
                os.path.join(self.cache_dir, "**", f"*{dataset_name_only}*", "**", "*.parquet"),  # 包含数据集名称的目录中的任何parquet文件
                os.path.join(self.cache_dir, "**", "*.parquet")  # 所有parquet文件
            ]
            
            for path_pattern in possible_paths:
                print(f"搜索路径: {path_pattern}")
                try:
                    found_files = glob.glob(path_pattern, recursive=True)
                    if found_files:
                        print(f"找到 {len(found_files)} 个可能的文件")
                        
                        # 检查每个文件，看是否可以用作当前索引的文件
                        for found_path in found_files:
                            found_name = os.path.basename(found_path)
                            found_dir = os.path.dirname(found_path)
                            
                            print(f"检查文件: {found_path}")
                            
                            # 检查文件是否为空
                            file_size = os.path.getsize(found_path)
                            if file_size == 0:
                                print(f"文件大小为0，跳过")
                                continue
                            
                            print(f"文件大小: {file_size / (1024*1024):.2f} MB")
                            
                            # 如果文件名已经是正确格式，但在非标准目录中
                            if found_name == file_name:
                                print(f"找到匹配的文件名: {found_path}")
                                
                                # 检查文件是否完整
                                try:
                                    if verify_parquet_file(found_path):
                                        print(f"文件验证通过")
                                        
                                        # 如果文件不在标准目录中，考虑复制或移动到标准目录
                                        target_dir = self.dir_formats[0]
                                        os.makedirs(target_dir, exist_ok=True)
                                        target_path = os.path.join(target_dir, file_name)
                                        
                                        if found_path != target_path:
                                            print(f"将文件复制到标准位置: {target_path}")
                                            shutil.copy2(found_path, target_path)
                                        
                                        return True, found_path
                                    else:
                                        print(f"文件验证失败")
                                except Exception as e:
                                    print(f"验证文件时出错: {e}")
                            
                            # 如果是其他格式的文件名，但可能是我们需要的文件
                            else:
                                # 检查是否为同一数据集的parquet文件
                                if dataset_name_only in found_path and found_path.endswith('.parquet'):
                                    print(f"找到可能相关的文件: {found_path}")
                                    
                                    # 验证文件是否完整
                                    try:
                                        if verify_parquet_file(found_path):
                                            print(f"文件验证通过，可能可以用作 {file_name}")
                                            
                                            # 复制到标准目录并重命名
                                            target_dir = self.dir_formats[0]
                                            os.makedirs(target_dir, exist_ok=True)
                                            target_path = os.path.join(target_dir, file_name)
                                            
                                            print(f"将文件复制并重命名为: {target_path}")
                                            shutil.copy2(found_path, target_path)
                                            
                                            return True, target_path
                                        else:
                                            print(f"文件验证失败")
                                    except Exception as e:
                                        print(f"验证文件时出错: {e}")
                except Exception as e:
                    print(f"搜索路径 {path_pattern} 时出错: {e}")
            
            print(f"文件 {file_name} 不存在，需要下载")
            return False, None
        
        def download_single_file(self, file_index):
            """下载单个parquet文件
            
            Args:
                file_index: 文件索引(0, 1, 2, ...)
                
            Returns:
                bool: 是否下载成功
            """
            file_name = f"{file_index:04d}.parquet"
            print(f"\n开始下载文件 {file_name}...")
            
            # 获取数据集信息，用于确定文件总数
            try:
                # 先判断文件是否已存在
                exists, file_path = self.check_file_exists(file_index)
                if exists:
                    print(f"文件 {file_name} 已存在且完整，跳过下载")
                    return True
                
                # 构建单个文件的下载参数
                # 注意: 这里使用自定义方法实现单文件下载，因为Hugging Face官方API不直接支持单文件下载
                print(f"尝试直接下载文件 {file_name}...")
                
                # 请求数据集的parquet文件列表
                parquet_url = f"https://huggingface.co/api/datasets/{dataset_name}/parquet"
                print(f"请求parquet文件列表: {parquet_url}")
                
                response = requests.get(parquet_url)
                if response.status_code == 200:
                    parquet_data = response.json()
                    
                    # 寻找对应split的文件URL
                    if self.dataset_name_only in parquet_data:
                        if self.split in parquet_data[self.dataset_name_only]:
                            file_urls = parquet_data[self.dataset_name_only][self.split]
                            
                            if file_index < len(file_urls):
                                file_url = file_urls[file_index]
                                print(f"找到文件URL: {file_url}")
                                
                                # 创建下载目录
                                download_dir = self.find_download_dir()
                                if not download_dir:
                                    download_dir = self.dir_formats[0]  # 使用第一种格式作为默认
                                    os.makedirs(download_dir, exist_ok=True)
                                    print(f"创建下载目录: {download_dir}")
                                
                                # 构建本地文件路径
                                local_file_path = os.path.join(download_dir, file_name)
                                
                                # 下载文件
                                print(f"下载文件到: {local_file_path}")
                                try:
                                    # 使用stream模式下载大文件
                                    with requests.get(file_url, stream=True) as r:
                                        r.raise_for_status()
                                        total_size = int(r.headers.get('content-length', 0))
                                        print(f"文件总大小: {total_size / (1024*1024):.2f} MB")
                                        
                                        with open(local_file_path, 'wb') as f:
                                            chunk_size = 8192  # 8KB
                                            downloaded = 0
                                            last_percent = -1
                                            
                                            for chunk in r.iter_content(chunk_size=chunk_size):
                                                if chunk:
                                                    f.write(chunk)
                                                    downloaded += len(chunk)
                                                    
                                                    # 显示下载进度
                                                    percent = int((downloaded / total_size) * 100)
                                                    if percent > last_percent:
                                                        print(f"下载进度: {percent}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                                                        last_percent = percent
                                    
                                    # 验证下载的文件
                                    print(f"验证下载的文件...")
                                    if verify_parquet_file(local_file_path):
                                        print(f"文件 {file_name} 下载成功并验证通过")
                                        return True
                                    else:
                                        print(f"文件 {file_name} 下载后验证失败")
                                        return False
                                
                                except Exception as e:
                                    print(f"下载文件时出错: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    return False
                            else:
                                print(f"请求的文件索引 {file_index} 超出可用文件数量 {len(file_urls)}")
                                return False
                        else:
                            print(f"在数据集中未找到分割 {self.split}")
                    else:
                        print(f"在parquet数据中未找到数据集 {self.dataset_name_only}")
                else:
                    print(f"请求parquet文件列表失败，状态码: {response.status_code}")
                
                # 如果直接下载失败，尝试使用官方API下载
                print(f"直接下载失败，尝试使用官方API下载整个数据集...")
                self.download_using_official_api()
                
                # 再次检查文件是否存在
                exists, file_path = self.check_file_exists(file_index)
                return exists
            
            except Exception as e:
                print(f"下载文件 {file_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        def download_using_official_api(self):
            """使用官方API下载整个数据集"""
            print(f"\n使用官方API下载数据集 {self.dataset_name}...")
            try:
                # 修改为直接使用HfApi下载各个文件
                print("尝试使用HfApi直接下载数据集文件...")
                
                # 获取下载路径格式
                download_dir = self.find_download_dir()
                if not download_dir:
                    download_dir = self.dir_formats[0]  # 使用第一种格式作为默认
                    os.makedirs(download_dir, exist_ok=True)
                    print(f"创建下载目录: {download_dir}")
                
                # 使用HfApi列出所有可用文件
                try:
                    api = HfApi()
                    print(f"获取数据集 {self.dataset_name} 的文件列表...")
                    dataset_files = api.list_repo_files(self.dataset_name, repo_type="dataset")
                    
                    # 过滤出当前split的parquet文件
                    parquet_files = [f for f in dataset_files if f.endswith('.parquet') and self.split in f]
                    
                    if parquet_files:
                        print(f"找到 {len(parquet_files)} 个可能的parquet文件")
                        # 对文件进行排序，确保按顺序下载
                        sorted_files = sorted(parquet_files)
                        print(f"文件将按以下顺序下载: {sorted_files}")
                        
                        # 开始下载文件
                        for i, file_path in enumerate(sorted_files):
                            # 构建本地文件名
                            local_filename = f"{i:04d}.parquet"
                            local_path = os.path.join(download_dir, local_filename)
                            
                            print(f"\n下载 {file_path} 到 {local_path}...")
                            try:
                                # 使用HfApi下载文件
                                hf_hub_download(
                                    repo_id=self.dataset_name,
                                    filename=file_path,
                                    repo_type="dataset",
                                    local_dir=download_dir,
                                    local_dir_use_symlinks=False,
                                    force_download=True
                                )
                                
                                # 文件可能下载到了不同的文件名，需要重命名
                                downloaded_path = os.path.join(download_dir, os.path.basename(file_path))
                                if os.path.exists(downloaded_path) and downloaded_path != local_path:
                                    print(f"文件下载成功，重命名为: {local_filename}")
                                    # 如果目标文件已存在，先删除
                                    if os.path.exists(local_path):
                                        os.remove(local_path)
                                    # 重命名文件
                                    shutil.move(downloaded_path, local_path)
                                
                                # 验证下载的文件
                                print(f"验证文件 {local_filename}...")
                                if os.path.exists(local_path):
                                    file_size = os.path.getsize(local_path)
                                    print(f"文件大小: {file_size / (1024*1024):.2f} MB")
                                    if verify_parquet_file(local_path):
                                        print(f"文件 {local_filename} 验证通过")
                                    else:
                                        print(f"文件 {local_filename} 验证失败，但已下载")
                                else:
                                    print(f"文件 {local_filename} 下载失败，未找到文件")
                            
                            except Exception as e:
                                print(f"下载文件 {file_path} 时出错: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # 检查下载结果
                        downloaded_files = [f for f in os.listdir(download_dir) if f.endswith('.parquet')]
                        print(f"\n下载完成，共下载了 {len(downloaded_files)} 个文件")
                        
                        return len(downloaded_files) > 0
                    else:
                        print(f"未找到符合条件的parquet文件")
                except Exception as e:
                    print(f"使用HfApi下载文件时出错: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 如果上面的方法失败，尝试使用官方的load_dataset方法
                print("\n尝试使用官方load_dataset API下载...")
                
                # 使用非流式下载，更可能下载实际文件
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=False,  # 必须为False才会下载
                    cache_dir=self.cache_dir,
                    download_mode="force_redownload"  # 强制重新下载
                )
                
                print(f"官方API下载完成，当前目录中的文件:")
                for dir_path in self.dir_formats:
                    if os.path.exists(dir_path):
                        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.parquet')])
                        if files:
                            print(f"目录 {dir_path} 中的文件:")
                            for f in files:
                                file_path = os.path.join(dir_path, f)
                                print(f"  - {f} (大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB)")
                        else:
                            print(f"目录 {dir_path} 中没有parquet文件")
                
                return True
            except Exception as e:
                print(f"使用官方API下载时出错: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        def sequential_download(self, max_files=100):
            """按顺序下载数据集的所有parquet文件
            
            Args:
                max_files: 最大尝试下载的文件数量，避免无限循环
                
            Returns:
                dataset: 加载的数据集对象
            """
            print(f"\n开始按顺序下载数据集 {self.dataset_name} 的parquet文件...")
            print(f"将按照 0000.parquet, 0001.parquet, ... 的顺序下载文件")
            print(f"系统会严格检查每个文件是否存在和完整，只有确认完整的文件才会跳过下载")
            
            # 找到下载目录
            download_dir = self.find_download_dir()
            
            # 尝试获取文件总数
            try:
                parquet_url = f"https://huggingface.co/api/datasets/{dataset_name}/parquet"
                response = requests.get(parquet_url)
                if response.status_code == 200:
                    parquet_data = response.json()
                    if self.dataset_name_only in parquet_data and self.split in parquet_data[self.dataset_name_only]:
                        file_urls = parquet_data[self.dataset_name_only][self.split]
                        print(f"数据集包含 {len(file_urls)} 个parquet文件")
                        max_files = min(max_files, len(file_urls))
                    else:
                        print(f"无法从API获取文件总数，将尝试下载最多 {max_files} 个文件")
                else:
                    print(f"无法从API获取文件总数，将尝试下载最多 {max_files} 个文件")
            except Exception as e:
                print(f"获取文件总数时出错: {e}")
                print(f"将尝试下载最多 {max_files} 个文件")
            
            # 按顺序下载文件
            file_index = 0
            consecutive_failures = 0
            
            print("\n===== 开始顺序下载过程 =====")
            print(f"如果使用官方API下载，请注意观察下载的文件名")
            
            while file_index < max_files and consecutive_failures < 3:
                print(f"\n===== 尝试下载文件 {file_index:04d}.parquet (第 {file_index+1} 个文件) =====")
                result = self.download_single_file(file_index)
                
                if result:
                    print(f"文件 {file_index:04d}.parquet 下载成功或已存在")
                    consecutive_failures = 0
                    file_index += 1
                else:
                    consecutive_failures += 1
                    print(f"文件 {file_index:04d}.parquet 下载失败，连续失败次数: {consecutive_failures}")
                    if consecutive_failures >= 3:
                        print(f"连续失败次数达到3次，停止下载")
                        break
                    
                    # 尝试跳过当前文件
                    file_index += 1
            
            # 下载完成后，尝试使用官方API加载数据集
            print(f"\n===== 所有文件下载尝试完成 =====")
            print(f"现在检查下载目录中的文件...")
            
            # 打印下载目录中的所有文件
            for dir_path in self.dir_formats:
                if os.path.exists(dir_path):
                    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.parquet')])
                    if files:
                        print(f"\n在目录 {dir_path} 中找到 {len(files)} 个parquet文件:")
                        for f in files:
                            file_path = os.path.join(dir_path, f)
                            size = os.path.getsize(file_path) / (1024*1024)
                            print(f"  - {f} (大小: {size:.2f} MB)")
                    else:
                        print(f"\n目录 {dir_path} 存在但没有parquet文件")
            
            print(f"\n使用官方API加载数据集...")
            try:
                dataset = load_dataset(
                    self.dataset_name,
                    split=self.split,
                    streaming=streaming,
                    cache_dir=self.cache_dir,
                    download_mode="reuse_dataset_if_exists"  # 直接使用已下载的文件
                )
                print(f"数据集加载成功!")
                return dataset
            except Exception as e:
                print(f"加载数据集时出错: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    # 创建下载器并开始顺序下载
    downloader = SequentialDownloader(dataset_name, split, cache_dir, verify_downloads=verify_downloads)
    dataset = downloader.sequential_download()
    
    # 下载后检查目录和文件
    print("\n下载完成后的目录检查:")
    for name_variant in [
        dataset_name.replace("/", "---"),
        dataset_name.replace("/", "--"),
        f"{downloader.dataset_org}--{downloader.dataset_name_only}" if downloader.dataset_org else downloader.dataset_name_only
    ]:
        possible_dir = os.path.join(cache_dir, name_variant, split)
        if os.path.exists(possible_dir):
            print(f"下载目录: {possible_dir}")
            files = sorted([f for f in os.listdir(possible_dir) if f.endswith('.parquet')])
            print(f"目录中的parquet文件: {len(files)} 个")
            for f in files:
                full_path = os.path.join(possible_dir, f)
                size = os.path.getsize(full_path)
                print(f"  - {f} (大小: {size / (1024*1024):.2f} MB)")
                
                # 验证文件完整性
                if verify_downloads:
                    try:
                        is_valid = verify_parquet_file(full_path)
                        print(f"  - 验证结果: {'通过' if is_valid else '失败'}")
                    except Exception as e:
                        print(f"  - 验证时出错: {e}")
    
    return dataset


def get_dataset_sample(dataset: Any, sample_size: int, streaming: bool = False) -> List[Dict]:
    """从数据集中获取样本
    
    Args:
        dataset: 数据集对象
        sample_size: 样本数量
        streaming: 是否为流式数据集
        
    Returns:
        数据样本列表
    """
    if streaming:
        samples = []
        for i, example in tqdm(enumerate(dataset), desc="采样数据", total=sample_size):
            if i >= sample_size:
                break
            samples.append(example)
        return samples
    else:
        if len(dataset) <= sample_size:
            return dataset
        indices = random.sample(range(len(dataset)), sample_size)
        return dataset.select(indices)


def analyze_text_length(samples: List[Dict], text_column: str, save_dir: str):
    """分析文本长度分布
    
    Args:
        samples: 数据样本
        text_column: 文本列名
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算文本长度
    text_lengths = [len(sample[text_column]) for sample in samples]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, kde=True)
    plt.title("文本长度分布")
    plt.xlabel("文本长度（字符数）")
    plt.ylabel("频率")
    plt.savefig(os.path.join(save_dir, "text_length_distribution.png"))
    plt.close()
    
    # 计算统计信息
    stats = {
        "平均长度": np.mean(text_lengths),
        "中位数长度": np.median(text_lengths),
        "最大长度": max(text_lengths),
        "最小长度": min(text_lengths),
        "标准差": np.std(text_lengths)
    }
    
    # 保存统计信息
    with open(os.path.join(save_dir, "text_length_stats.txt"), "w", encoding="utf-8") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print("文本长度分析完成，结果已保存到", save_dir)
    return stats


def analyze_word_frequency(samples: List[Dict], text_column: str, save_dir: str, top_n: int = 50):
    """分析词频
    
    Args:
        samples: 数据样本
        text_column: 文本列名
        save_dir: 保存目录
        top_n: 展示前N个高频词
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 分词并统计词频
    all_words = []
    for sample in tqdm(samples, desc="分词并统计词频"):
        words = jieba.lcut(sample[text_column])
        all_words.extend(words)
    
    # 过滤掉停用词和单个字符
    filtered_words = [word for word in all_words if len(word) > 1]
    
    # 统计词频
    word_counts = Counter(filtered_words)
    most_common = word_counts.most_common(top_n)
    
    # 绘制词频柱状图
    words, counts = zip(*most_common)
    plt.figure(figsize=(15, 8))
    plt.bar(words, counts)
    plt.title(f"Top {top_n} 高频词")
    plt.xlabel("词语")
    plt.ylabel("频率")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "word_frequency.png"))
    plt.close()
    
    # 保存词频信息
    with open(os.path.join(save_dir, "word_frequency.txt"), "w", encoding="utf-8") as f:
        for word, count in most_common:
            f.write(f"{word}: {count}\n")
    
    print("词频分析完成，结果已保存到", save_dir)


def display_random_examples(samples: List[Dict], text_column: str, n: int = 5):
    """显示随机示例
    
    Args:
        samples: 数据样本
        text_column: 文本列名
        n: 显示的示例数量
    """
    random_samples = random.sample(samples, min(n, len(samples)))
    print(f"\n随机{n}个示例:")
    for i, sample in enumerate(random_samples, 1):
        text = sample[text_column]
        # 截断过长的文本
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"示例 {i}:\n{text}\n")


def visualize_dataset_schema(samples: List[Dict], save_dir: str):
    """可视化数据集结构
    
    Args:
        samples: 数据样本
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取第一个样本，分析其结构
    if not samples:
        print("没有可用的样本来分析数据集结构")
        return
    
    sample = samples[0]
    schema = {}
    
    for key, value in sample.items():
        if isinstance(value, (str, int, float, bool)):
            schema[key] = type(value).__name__
        elif isinstance(value, list):
            schema[key] = f"list[{type(value[0]).__name__ if value else 'unknown'}]"
        elif isinstance(value, dict):
            schema[key] = "dict"
        else:
            schema[key] = str(type(value).__name__)
    
    # 保存结构信息
    with open(os.path.join(save_dir, "dataset_schema.txt"), "w", encoding="utf-8") as f:
        f.write("数据集结构:\n")
        for key, type_name in schema.items():
            f.write(f"{key}: {type_name}\n")
    
    print("数据集结构分析完成，结果已保存到", save_dir)


def main():
    """主函数"""
    # 确保缓存目录存在
    for cache_path in [
        os.environ["HF_HOME"],
        os.environ["HUGGINGFACE_HUB_CACHE"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["HF_DATASETS_CACHE"]
    ]:
        os.makedirs(cache_path, exist_ok=True)
    
    # 显示当前缓存路径
    show_cache_paths()
    
    args = setup_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 下载数据集
    dataset = download_dataset(
        args.dataset, 
        args.split, 
        args.streaming,
        args.cache_dir,
        args.force_download,
        args.verify_downloads
    )
    
    if args.download_only:
        print("数据集下载完成。因为指定了--download_only，所以跳过可视化步骤。")
        return
    
    # 获取样本
    samples = get_dataset_sample(dataset, args.sample_size, args.streaming)
    
    # 显示数据集信息
    if not args.streaming:
        print(f"数据集大小: {len(dataset)} 条记录")
    print(f"样本大小: {len(samples)} 条记录")
    
    # 显示随机示例
    display_random_examples(samples, args.text_column)
    
    # 分析数据集结构
    visualize_dataset_schema(samples, args.save_dir)
    
    # 分析文本长度
    stats = analyze_text_length(samples, args.text_column, args.save_dir)
    print("文本长度统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 分析词频
    analyze_word_frequency(samples, args.text_column, args.save_dir)
    
    print(f"\n可视化结果已保存到: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main() 