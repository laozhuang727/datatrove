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
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
from huggingface_hub.utils import validate_repo_id
from datasets.utils.file_utils import get_datasets_user_agent
from pathlib import Path
import tempfile
import time

# # 设置Hugging Face的缓存目录
# os.environ["HF_HOME"] = r"D:\huggingface_cache"
# os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\huggingface_cache\hub"
# os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface_cache\transformers"
# os.environ["HF_DATASETS_CACHE"] = r"D:\huggingface_cache\datasets"

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
    parser.add_argument("--local_dir", type=str, default=None,
                       help="下载文件到本地指定目录")
    return parser.parse_args()


def show_cache_paths():
    """显示当前Hugging Face缓存路径"""
    print("\n当前Hugging Face缓存路径:")
    print(f"HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
    print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', '未设置')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', '未设置')}")
    print()


def verify_parquet_file(file_path: str) -> bool:
    """验证parquet文件的完整性
    
    Args:
        file_path: parquet文件路径
        
    Returns:
        bool: 文件是否完整
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    try:
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


def download_dataset(dataset_name: str, split: str, streaming: bool = False, 
                    cache_dir: Optional[str] = None, force_download: bool = False,
                    verify_downloads: bool = False, local_dir: Optional[str] = None) -> Any:
    """下载指定的Hugging Face数据集
    
    使用官方的huggingface_hub库下载数据集文件
    
    Args:
        dataset_name: 数据集名称
        split: 数据集分割
        streaming: 是否流式加载
        cache_dir: 缓存目录
        force_download: 是否强制重新下载
        verify_downloads: 是否验证下载文件完整性
        local_dir: 本地保存目录
        
    Returns:
        加载的数据集对象
    """
    print(f"正在下载数据集: {dataset_name}, 分割: {split}")
    
    # 如果未指定cache_dir，使用环境变量设置的路径
    if cache_dir is None:
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
        if not cache_dir:
            print(f"警告: 未指定缓存目录且环境变量HF_DATASETS_CACHE未设置")
            print(f"数据集可能会下载到默认位置(用户主目录下的.cache/huggingface)")
            print(f"如需指定缓存目录，请使用--cache_dir参数或设置HF_DATASETS_CACHE环境变量")
        else:
            print(f"使用环境变量中的缓存目录: {cache_dir}")
    else:
        print(f"使用命令行指定的缓存目录: {cache_dir}")
        # 确保目录存在
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"已确认缓存目录存在: {cache_dir}")
        except Exception as e:
            print(f"警告: 无法创建缓存目录 {cache_dir}: {e}")
    
    # 使用local_dir时的提示
    if local_dir:
        print(f"将下载文件保存到本地目录: {local_dir}")
        try:
            os.makedirs(local_dir, exist_ok=True)
            print(f"已确认本地目录存在: {local_dir}")
        except Exception as e:
            print(f"警告: 无法创建本地目录 {local_dir}: {e}")
    
    # 1. 使用HfApi获取数据集文件列表
    try:
        api = HfApi()
        print(f"获取数据集 {dataset_name} 的文件列表...")
        dataset_files = api.list_repo_files(dataset_name, repo_type="dataset")
        print(f"找到 {len(dataset_files)} 个文件")
        
        # 过滤出与分割相关的parquet文件
        split_files = [f for f in dataset_files if split in f and f.endswith('.parquet')]
        print(f"当前分割 '{split}' 包含 {len(split_files)} 个parquet文件")
        
        if not split_files:
            print(f"未找到与分割 '{split}' 相关的parquet文件，将尝试使用load_dataset API")
        else:
            print(f"\n===== 开始下载parquet文件 =====")
            # 2. 下载单个文件的情况
            downloaded_files = []
            
            for i, file_path in enumerate(sorted(split_files), 1):
                print(f"\n[{i}/{len(split_files)}] 下载文件: {file_path}")
                
                try:
                    # 使用官方的hf_hub_download来下载文件
                    download_params = {
                        "repo_id": dataset_name,
                        "filename": file_path,
                        "repo_type": "dataset",
                        "force_download": force_download
                    }
                    
                    # 只有在指定了参数时才添加到下载参数中
                    if cache_dir:
                        download_params["cache_dir"] = cache_dir
                    if local_dir:
                        download_params["local_dir"] = local_dir
                    
                    # 执行下载
                    print(f"下载参数: {download_params}")
                    downloaded_file = hf_hub_download(**download_params)
                    
                    print(f"文件下载成功: {file_path}")
                    print(f"【下载路径】: {downloaded_file}")
                    print(f"【绝对路径】: {os.path.abspath(downloaded_file)}")
                    
                    # 验证下载的文件
                    if verify_downloads:
                        print(f"正在验证文件完整性...")
                        if verify_parquet_file(downloaded_file):
                            print(f"文件验证通过")
                        else:
                            print(f"警告：文件 {file_path} 验证失败，可能损坏")
                    
                    downloaded_files.append(downloaded_file)
                except Exception as e:
                    print(f"下载文件 {file_path} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 下载完成后总结
            print("\n===== 下载完成 =====")
            print(f"成功下载 {len(downloaded_files)}/{len(split_files)} 个文件")
            
            # 打印所有下载的文件
            if downloaded_files:
                print("\n===== 已下载的文件列表 =====")
                for i, file_path in enumerate(downloaded_files, 1):
                    print(f"{i}. {os.path.basename(file_path)}")
                    print(f"   【完整路径】: {os.path.abspath(file_path)}")
                    print(f"   【文件大小】: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            
            # 3. 尝试使用snapshot_download下载整个仓库
            if not downloaded_files or len(downloaded_files) < len(split_files):
                print("\n部分文件下载失败，尝试使用snapshot_download下载整个分割...")
                try:
                    # 使用allow_patterns参数只下载特定分割的文件
                    download_params = {
                        "repo_id": dataset_name,
                        "repo_type": "dataset",
                        "allow_patterns": f"*{split}*.parquet",
                        "force_download": force_download
                    }
                    
                    # 只有在指定了参数时才添加到下载参数中
                    if cache_dir:
                        download_params["cache_dir"] = cache_dir
                    if local_dir:
                        download_params["local_dir"] = local_dir
                    
                    print(f"快照下载参数: {download_params}")
                    snapshot_path = snapshot_download(**download_params)
                    
                    print(f"分割快照下载完成")
                    print(f"【快照路径】: {snapshot_path}")
                    print(f"【绝对路径】: {os.path.abspath(snapshot_path)}")
                    
                    # 查找下载的parquet文件
                    pattern = os.path.join(snapshot_path, "**", f"*{split}*.parquet")
                    found_files = glob.glob(pattern, recursive=True)
                    
                    if found_files:
                        print(f"在快照中找到 {len(found_files)} 个parquet文件")
                        for i, file_path in enumerate(found_files, 1):
                            print(f"{i}. {os.path.basename(file_path)}")
                            print(f"   【完整路径】: {os.path.abspath(file_path)}")
                            print(f"   【文件大小】: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
                            downloaded_files.append(file_path)
                except Exception as e:
                    print(f"使用snapshot_download下载时出错: {e}")
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(f"获取数据集文件列表时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 使用官方datasets库加载数据集
    print(f"\n使用datasets库加载数据集: {dataset_name}...")
    try:
        # 提供详细信息
        print(f"缓存目录参数: {cache_dir}")
        print(f"分割: {split}")
        print(f"流式加载: {streaming}")
        print(f"下载模式: {'force_redownload' if force_download else 'reuse_dataset_if_exists'}")
        
        # 执行加载
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
            download_mode="force_redownload" if force_download else "reuse_dataset_if_exists"
        )
        
        print(f"数据集加载成功!")
        
        # 5. 如果使用datasets库加载成功，找出实际下载的文件
        print("\n查找datasets库下载的文件位置...")
        
        # 构建搜索模式
        all_patterns = []
        
        # 如果有缓存目录，在缓存目录中查找
        if cache_dir:
            all_patterns.extend([
                # 标准格式
                os.path.join(cache_dir, "**", dataset_name.replace("/", "--"), "**", f"*{split}*.parquet"),
                os.path.join(cache_dir, "**", dataset_name.replace("/", "---"), "**", f"*{split}*.parquet"),
                # 一般格式
                os.path.join(cache_dir, "**", f"*{dataset_name.split('/')[-1]}*", "**", "*.parquet")
            ])
        
        # 在默认位置查找
        default_paths = [
            os.path.expanduser("~/.cache/huggingface/datasets"),
            os.path.join(os.getcwd(), ".cache/huggingface/datasets")
        ]
        
        for default_path in default_paths:
            all_patterns.extend([
                os.path.join(default_path, "**", dataset_name.replace("/", "--"), "**", f"*{split}*.parquet"),
                os.path.join(default_path, "**", dataset_name.replace("/", "---"), "**", f"*{split}*.parquet")
            ])
        
        # 如果指定了local_dir，也在那里查找
        if local_dir:
            all_patterns.append(os.path.join(local_dir, "**", "*.parquet"))
        
        # 开始搜索
        found_files = []
        for pattern in all_patterns:
            print(f"搜索: {pattern}")
            try:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    print(f"  找到 {len(matches)} 个匹配文件")
                    found_files.extend(matches)
            except Exception as e:
                print(f"  搜索出错: {e}")
        
        # 去重
        found_files = list(set(found_files))
        
        # 打印结果
        if found_files:
            print(f"\n找到 {len(found_files)} 个datasets库缓存的文件")
            for i, file_path in enumerate(sorted(found_files), 1):
                print(f"{i}. {os.path.basename(file_path)}")
                print(f"   【完整路径】: {os.path.abspath(file_path)}")
                print(f"   【文件大小】: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
                
                # 验证文件存在并打印最后修改时间
                if os.path.exists(file_path):
                    mtime = os.path.getmtime(file_path)
                    print(f"   【最后修改】: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
        else:
            print("未找到datasets库缓存的文件")
            
            # 尝试从数据集对象获取信息
            if not streaming and hasattr(dataset, '_data_files') and dataset._data_files:
                print("\n尝试从数据集对象获取文件信息:")
                for split_name, file_list in dataset._data_files.items():
                    print(f"  分割 {split_name}:")
                    for file_path in file_list:
                        print(f"    - {file_path}")
                        if os.path.exists(file_path):
                            print(f"      【存在】: {os.path.abspath(file_path)}")
                            print(f"      【大小】: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
                            print(f"      【最后修改】: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_path)))}")
        
        return dataset
    
    except Exception as e:
        print(f"使用datasets库加载数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        raise


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
    # 只检查HF_HOME环境变量是否已设置
    if "HF_HOME" not in os.environ or not os.environ["HF_HOME"]:
        print("\n警告：HF_HOME环境变量未设置。Hugging Face缓存可能会使用默认位置（用户主目录下的.cache）")
        print("请设置HF_HOME环境变量指定Hugging Face的缓存根目录。")
        print("\n在Windows上可以使用:")
        print(r'set HF_HOME=D:\huggingface_cache')
        print("\n或者在Python脚本开头添加:")
        print(r'os.environ["HF_HOME"] = r"D:\huggingface_cache"')
        
        print("\n是否继续执行脚本？这可能导致文件下载到默认位置（可能是用户主目录下的.cache文件夹）")
        response = input("继续执行？(y/n): ")
        if response.lower() != 'y':
            print("脚本已终止。请设置HF_HOME环境变量后重新运行。")
            return
    else:
        # 确保HF_HOME目录存在
        hf_home = os.environ["HF_HOME"]
        try:
            os.makedirs(hf_home, exist_ok=True)
            print(f"已确认HF_HOME目录存在: {hf_home}")
        except Exception as e:
            print(f"警告: 无法创建HF_HOME目录 {hf_home}: {e}")
    
    # 显示当前缓存路径
    show_cache_paths()
    
    args = setup_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # 下载数据集
        dataset = download_dataset(
            args.dataset, 
            args.split, 
            args.streaming,
            args.cache_dir,
            args.force_download,
            args.verify_downloads,
            args.local_dir
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
    
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 