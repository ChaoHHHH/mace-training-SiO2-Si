"""
MACE 模型训练脚本
==================
基于 T01_MACE_Practice_I.ipynb 教程实现。
读取 .xyz 数据文件，训练 MACE 势函数模型。

用法: 直接修改下方 CONFIG，然后运行
    python train_mace_from_xyz.py

依赖:
    pip install mace-torch ase torch
"""

import os
import sys
import logging
import warnings
import re
import yaml
import numpy as np
from ase.io import read
from mace.cli.run_train import main as mace_run_train_main

warnings.filterwarnings("ignore")

# ===========================================================================
#                          配置区 (修改这里即可)
# ===========================================================================

CONFIG = {
    # ----- 数据 -----
    # 方式一: 直接指定已拆分好的文件
    "train_file": None,                                    # 训练集路径 (split_source 不为空时自动生成)
    "test_file": None,                                     # 测试集路径 (split_source 不为空时自动生成)
    "valid_fraction": 0.0,                                 # 从训练集切分验证集的比例 (有 valid_file 时设为 0)

    # 方式二: 从单个文件自动拆分 (推荐)
    "split_source": "example.xyz",                       # 原始 .xyz 文件, 设为 None 则使用上面的 train_file
    "split_ratios": [0.8, 0.1, 0.1],                       # [训练, 验证, 测试] 比例
    "split_seed": 42,                                      # 拆分随机种子
    "split_output_dir": "splitted_data",                   # 拆分后文件的输出目录

    "energy_key": "energy",                              # 能量字段名
    "forces_key": "forces",                              # 力字段名
    "E0s": "average",                                    # 原子能量: "average" 或 "isolated"
    
    # ----- 模型架构 -----
    "num_channels": 128,            # 通道数 (64快速, 128默认, 256高精度)
    "max_L": 1,                    # 消息对称性 (0最快, 1推荐, 2最精确)
    "r_max": 5.0,                  # 截断半径 Å (推荐 4.0~7.0)
    "max_ell": 3,                  # 球谐函数最大阶数
    "correlation": 3,              # 多体展开阶数
    "num_interactions": 2,         # 消息传递层数 (保持2)
    "num_radial_basis": 8,         # 径向基函数数量

    # ----- 训练控制 -----
    "name": "mace_model",       # 模型名称
    "output_dir": "MACE_models",   # 输出目录
    "device": "cuda",              # 设备: "auto" / "cuda" / "cpu" / "mps"
    "batch_size": 10,              # 批大小
    "max_num_epochs": 500,         # 最大训练轮数
    "seed": 42,                   # 随机种子
    "start_swa": 200,               # SWA 阶段起始 epoch
    "patience": 2048,              # 早停耐心
    
    # ----- 损失权重 -----
    "lr": 0.01,                    # 初始学习率
    "swa_lr": 0.001,               # SWA 阶段学习率
    "energy_weight": 1.0,          # 能量损失权重
    "forces_weight": 100.0,        # 力损失权重
    "swa_energy_weight": 1000.0,   # SWA 阶段能量权重
}

# ===========================================================================
#                           以下无需修改
# ===========================================================================


def determine_device(device_arg):
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def create_yaml_config(cfg):
    """生成 MACE 训练 YAML 配置"""
    config = {
        "model": "MACE",
        "num_channels": cfg["num_channels"],
        "max_L": cfg["max_L"],
        "r_max": cfg["r_max"],
        "max_ell": cfg["max_ell"],
        "correlation": cfg["correlation"],
        "num_interactions": cfg["num_interactions"],
        "num_radial_basis": cfg["num_radial_basis"],
        "name": cfg["name"],
        "model_dir": cfg["output_dir"],
        "log_dir": cfg["output_dir"],
        "checkpoints_dir": cfg["output_dir"],
        "results_dir": cfg["output_dir"],
        "train_file": os.path.abspath(cfg["train_file"]),
        "valid_fraction": cfg["valid_fraction"],
        "energy_key": cfg["energy_key"],
        "forces_key": cfg["forces_key"],
        "E0s": cfg["E0s"],
        "device": determine_device(cfg["device"]),
        "batch_size": cfg["batch_size"],
        "max_num_epochs": cfg["max_num_epochs"],
        "seed": cfg["seed"],
        "lr": cfg["lr"],
        "swa_lr": cfg["swa_lr"],
        "start_swa": cfg["start_swa"],
        "patience": cfg["patience"],
        "forces_weight": cfg["forces_weight"],
        "energy_weight": cfg["energy_weight"],
        "swa_energy_weight": cfg["swa_energy_weight"],
        "swa": True,
        "ema": True,
        "ema_decay": 0.99,
        "optimizer": "adam",
        "weight_decay": 5e-7,
        "amsgrad": True,
        "scheduler": "ReduceLROnPlateau",
        "lr_factor": 0.8,
        "scheduler_patience": 50,
        "clip_grad": 10.0,
    }
    if cfg.get("test_file"):
        config["test_file"] = os.path.abspath(cfg["test_file"])
    if cfg.get("valid_file"):
        config["valid_file"] = os.path.abspath(cfg["valid_file"])
        config["valid_fraction"] = 0.0
    return config


def analyze_xyz(file_path):
    """分析 .xyz 文件的基本信息"""
    data = read(file_path, ":")
    with open(file_path, "r") as f:
        lines = f.readlines()

    print(f"\n{'='*50}")
    print(f"  数据文件分析: {file_path}")
    print(f"{'='*50}")
    
    # 原始注释头
    n1 = int(lines[0].strip())
    print(f"结构1 注释头: {lines[1].strip()}")
    if len(lines) > n1 + 3:
        n2 = int(lines[n1 + 2].strip())
        if len(lines) > n1 + n2 + 3:
            print(f"结构2 注释头: {lines[n1 + 3].strip()}")
    
    print(f"总结构数: {len(data)}")
    
    all_species = set()
    for atoms in data:
        all_species.update(atoms.get_chemical_symbols())
    print(f"原子种类: {sorted(all_species)}")
    
    n_atoms_list = [len(atoms) for atoms in data]
    print(f"原子数范围: {min(n_atoms_list)} ~ {max(n_atoms_list)}")
    
    # ASE 解析结果
    sample = data[0]
    print(f"\nASE info.keys():  {list(sample.info.keys())}")
    print(f"ASE arrays.keys(): {list(sample.arrays.keys())}")
    
    # 检查 calc
    all_calc_keys = set()
    for atoms in data[:5]:
        if hasattr(atoms, "calc") and atoms.calc is not None:
            try:
                if atoms.get_potential_energy() is not None:
                    all_calc_keys.add("energy (from calc)")
            except Exception:
                pass
            try:
                if atoms.get_forces() is not None:
                    all_calc_keys.add("forces (from calc)")
            except Exception:
                pass
    if all_calc_keys:
        print(f"ASE calc:         {sorted(all_calc_keys)}")
    
    # 从原始文件检测字段
    print(f"\n--- 从原始文件检测 ---")
    column_keys = set()
    header_keys = set()
    line_idx = 0
    struct_count = 0
    while line_idx < len(lines) and struct_count < 5:
        try:
            n_atoms = int(lines[line_idx].strip())
        except ValueError:
            line_idx += 1
            continue
        comment = lines[line_idx + 1]
        props_match = re.search(r'Properties=([^\s]+)', comment)
        if props_match:
            for part in props_match.group(1).split(":"):
                if ":" in part:
                    name = part.split(":")[0]
                    if name not in ("species", "pos"):
                        column_keys.add(name)
        for match in re.finditer(r'(\w+)=(?:"[^"]*"|[^\s"]+)', comment):
            k = match.group(1)
            if k not in ("Properties", "Lattice"):
                header_keys.add(k)
        line_idx += n_atoms + 2
        struct_count += 1
    
    print(f"数据列:     {sorted(column_keys)}")
    print(f"key=value:  {sorted(header_keys)}")
    print(f"{'='*50}\n")
    
    return data


def split_dataset(cfg):
    """
    从单个 .xyz 文件拆分为训练/验证/测试集。
    - 自动识别 config_type=IsolatedAtom 的结构，始终放入训练集。
    - 其余结构按 cfg["split_ratios"] 随机打乱拆分。
    - 输出 train.xyz, valid.xyz, test.xyz 到 split_output_dir。
    - 返回更新后的 train_file, test_file，以及 valid_file 路径。
    """
    source = cfg["split_source"]
    ratios = cfg["split_ratios"]
    output_dir = cfg["split_output_dir"]

    if not os.path.exists(source):
        print(f"[ERROR] split_source 文件不存在: {source}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  拆分数据集: {source}")
    print(f"{'='*50}")

    data = read(source, ":")
    total = len(data)

    # 分离 IsolatedAtom 和普通结构
    isolated = []
    normal = []
    for atoms in data:
        ct = atoms.info.get("config_type", "")
        if ct == "IsolatedAtom":
            isolated.append(atoms)
        else:
            normal.append(atoms)

    print(f"总结构数:       {total}")
    print(f"  IsolatedAtom: {len(isolated)}")
    print(f"  普通结构:     {len(normal)}")

    if len(normal) < 3:
        print("[ERROR] 普通结构数量不足，无法拆分 (至少需要 3 个)")
        sys.exit(1)

    # 随机打乱普通结构
    import random
    rng = random.Random(cfg["split_seed"])
    rng.shuffle(normal)

    # 按比例拆分
    r_train, r_val, r_test = ratios
    n_total = len(normal)
    n_train = int(n_total * r_train)
    n_val = int(n_total * r_val)
    # 剩余归测试集，避免四舍五入丢失
    n_test = n_total - n_train - n_val

    train_data = normal[:n_train] + isolated        # IsolatedAtom 始终加入训练集
    valid_data = normal[n_train:n_train + n_val]
    test_data  = normal[n_train + n_val:]

    print(f"拆分比例:       {ratios}")
    print(f"  训练集 (含IsolatedAtom): {len(train_data)}")
    print(f"  验证集:                 {len(valid_data)}")
    print(f"  测试集:                 {len(test_data)}")

    # 写入文件
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.xyz")
    valid_path = os.path.join(output_dir, "valid.xyz")
    test_path  = os.path.join(output_dir, "test.xyz")

    from ase.io import write
    write(train_path, train_data)
    write(valid_path, valid_data)
    write(test_path,  test_data)

    print(f"\n已写入:")
    print(f"  {os.path.abspath(train_path)}")
    print(f"  {os.path.abspath(valid_path)}")
    print(f"  {os.path.abspath(test_path)}")
    print(f"{'='*50}\n")

    return train_path, valid_path, test_path


def check_keys(cfg, sample):
    """检查 energy/forces key 是否可用"""
    energy_ok = (cfg["energy_key"] in sample.info or 
                 cfg["energy_key"] in sample.arrays)
    if not energy_ok:
        try:
            if sample.get_potential_energy() is not None:
                energy_ok = True
        except Exception:
            pass
    
    forces_ok = (cfg["forces_key"] in sample.info or 
                 cfg["forces_key"] in sample.arrays)
    if not forces_ok:
        try:
            if sample.get_forces() is not None:
                forces_ok = True
        except Exception:
            pass
    
    if not energy_ok:
        print(f"[ERROR] energy_key='{cfg['energy_key']}' 未找到!")
        print(f"        请在 CONFIG 中修改 energy_key")
        return False
    if not forces_ok:
        print(f"[ERROR] forces_key='{cfg['forces_key']}' 未找到!")
        print(f"        请在 CONFIG 中修改 forces_key")
        return False
    return True


def main():
    cfg = CONFIG

    # 0. 如果配置了 split_source，先拆分数据
    if cfg.get("split_source"):
        train_file, valid_file, test_file = split_dataset(cfg)
        cfg = dict(cfg)  # 复制一份，避免修改全局 CONFIG
        cfg["train_file"] = train_file
        cfg["test_file"] = test_file
        cfg["valid_file"] = valid_file
        cfg["valid_fraction"] = 0.0  # 已有独立验证集
    else:
        cfg["valid_file"] = None

    # 1. 检查文件
    if not os.path.exists(cfg["train_file"]):
        print(f"[ERROR] 训练文件不存在: {cfg['train_file']}")
        sys.exit(1)
    
    # 2. 分析数据
    data = analyze_xyz(cfg["train_file"])
    
    # 3. 检查 key
    if not check_keys(cfg, data[0]):
        sys.exit(1)
    
    # 4. 创建输出目录
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # 5. 生成 YAML 并保存
    yaml_config = create_yaml_config(cfg)
    config_path = os.path.join(cfg["output_dir"], f"{cfg['name']}_config.yml")
    with open(config_path, "w") as f:
        yaml.dump(yaml_config, f, default_flow_style=False)
    print(f"[INFO] 配置文件: {config_path}")
    
    # 6. 打印摘要
    print(f"\n{'='*50}")
    print(f"  训练配置")
    print(f"{'='*50}")
    print(f"模型:     {cfg['name']}")
    print(f"设备:     {yaml_config['device']}")
    print(f"训练文件: {cfg['train_file']}")
    val_info = cfg.get("valid_file")
    if not val_info:
        val_info = f"{cfg['valid_fraction'] * 100:.0f}% from train"
    print(f"验证文件: {val_info}")
    print(f"测试文件: {cfg['test_file'] or '无'}")
    print(f"通道数:   {cfg['num_channels']}  |  max_L: {cfg['max_L']}  |  r_max: {cfg['r_max']}")
    print(f"epochs:   {cfg['max_num_epochs']}  |  batch: {cfg['batch_size']}")
    print(f"energy_key: {cfg['energy_key']}  |  forces_key: {cfg['forces_key']}")
    print(f"输出目录: {os.path.abspath(cfg['output_dir'])}")
    print(f"{'='*50}\n")
    
    # 7. 训练
    print("[INFO] 开始训练...\n")
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_path]
    mace_run_train_main()
    
    print(f"\n[INFO] 完成! 模型保存在: {os.path.abspath(cfg['output_dir'])}/")


if __name__ == "__main__":
    main()
