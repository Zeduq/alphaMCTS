"""
生成 Alpha101 因子数据集

从 trade-learn-master 中的 alphas101.py 导入因子计算公式，
基于沪深300日频交易数据计算101个Alpha因子，生成alpha101.csv。

使用方法:
    python generate_alpha101.py

输入数据格式要求:
    - 数据文件: D:/AAProject/Data/000300SH.csv
    - 必需字段: date, code, open, high, low, close, volume, vwap
    - 数据格式: CSV，长格式（每行一个股票一天的记录）

输出:
    - 文件: D:/AAProject/Data/alpha101.csv
    - 格式: 宽格式（date, code, alpha001_101, alpha002_101, ...）
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入 trade-learn 的 Alpha101 类
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trade-learn-master'))
from tradelearn.query.alpha.alphas101 import Alphas101

# 配置
DATA_PATH = 'D:/AAProject/Data/000300SH.csv'
OUTPUT_PATH = 'D:/AAProject/Data/alpha101_generated.csv'


def load_hs300_data(data_path: str) -> pd.DataFrame:
    """
    加载沪深300数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        DataFrame with columns: date, code, open, high, low, close, volume, vwap
    """
    print(f"正在加载数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 读取数据
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # 显示原始列名
    print(f"原始数据列名: {df.columns.tolist()}")
    print(f"原始数据形状: {df.shape}")
    
    # 确保列名正确（转换为小写以便匹配）
    df.columns = [col.lower() for col in df.columns]
    
    # 检查必需字段
    required_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'vwap']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # 尝试从其他列推断
        if 'vwap' not in df.columns and 'avg_price' in df.columns:
            df['vwap'] = df['avg_price']
            print("使用 'avg_price' 作为 'vwap'")
        elif 'vwap' not in df.columns:
            # 如果没有vwap，用 (high + low + close) / 3 近似
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
            print("使用 (high+low+close)/3 近似 vwap")
        
        # 再次检查
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需字段: {missing_cols}")
    
    # 选择并重命名列
    df = df[required_cols].copy()
    
    # 确保数据类型正确
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除包含NaN的行
    initial_len = len(df)
    df = df.dropna(subset=numeric_cols)
    if len(df) < initial_len:
        print(f"删除了 {initial_len - len(df)} 行包含NaN的数据")
    
    print(f"处理后数据形状: {df.shape}")
    print(f"数据日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"股票数量: {df['code'].nunique()}")
    
    return df


def prepare_data_for_alphas(df: pd.DataFrame) -> dict:
    """
    将长格式数据转换为宽格式字典，适配 Alphas101 类
    
    Args:
        df: 长格式DataFrame (date, code, open, high, low, close, volume, vwap)
        
    Returns:
        dict: 包含宽格式DataFrame的字典，键为字段名
    """
    print("\n正在转换数据格式...")
    
    # 将长格式转为宽格式（日期为索引，股票代码为列）
    pivot_data = {}
    fields = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    
    for field in fields:
        pivot_df = df.pivot(index='date', columns='code', values=field)
        pivot_data[field] = pivot_df
        print(f"  {field}: 形状 {pivot_df.shape}")
    
    return pivot_data


def calculate_alpha_factors(pivot_data: dict) -> pd.DataFrame:
    """
    计算所有Alpha101因子
    
    Args:
        pivot_data: 宽格式数据字典
        
    Returns:
        DataFrame: 因子值（长格式：date, code, alpha001, alpha002, ...）
    """
    print("\n开始计算Alpha101因子...")
    
    # 创建Alphas101实例
    alpha_calc = Alphas101(pivot_data)
    
    # 获取所有alpha方法
    alpha_methods = [method for method in dir(alpha_calc) 
                     if method.startswith('alpha') and method[5:].isdigit()]
    alpha_methods.sort(key=lambda x: int(x[5:]))
    
    print(f"发现 {len(alpha_methods)} 个Alpha因子方法")
    
    # 存储所有因子结果
    all_factors = {}
    failed_alphas = []
    
    for method_name in alpha_methods:
        alpha_num = int(method_name[5:])
        col_name = f'alpha{alpha_num:03d}_101'
        
        try:
            # 调用alpha计算方法
            alpha_func = getattr(alpha_calc, method_name)
            result = alpha_func()
            
            # 检查结果是否有效
            if result is not None and not result.empty:
                all_factors[col_name] = result
                print(f"  ✓ {col_name} 计算成功")
            else:
                failed_alphas.append(col_name)
                print(f"  ✗ {col_name} 返回空值")
                
        except Exception as e:
            failed_alphas.append(col_name)
            print(f"  ✗ {col_name} 计算失败: {str(e)[:50]}")
    
    print(f"\n成功计算: {len(all_factors)} 个因子")
    print(f"计算失败: {len(failed_alphas)} 个因子")
    if failed_alphas:
        print(f"  失败的因子: {', '.join(failed_alphas[:10])}{'...' if len(failed_alphas) > 10 else ''}")
    
    # 合并所有因子结果
    if not all_factors:
        raise ValueError("没有成功计算任何因子")
    
    # 将所有因子从宽格式（date x code）转为长格式（date, code, factor_value）
    print("\n正在合并因子数据...")
    
    # 使用第一个因子作为基础
    first_factor = list(all_factors.keys())[0]
    base_df = all_factors[first_factor].stack().reset_index()
    base_df.columns = ['date', 'code', first_factor]
    
    # 合并其他因子
    for col_name, factor_df in list(all_factors.items())[1:]:
        factor_long = factor_df.stack().reset_index()
        factor_long.columns = ['date', 'code', col_name]
        base_df = base_df.merge(factor_long, on=['date', 'code'], how='outer')
    
    # 按日期和股票代码排序
    base_df = base_df.sort_values(['date', 'code']).reset_index(drop=True)
    
    print(f"最终数据形状: {base_df.shape}")
    
    return base_df


def save_alpha101(df: pd.DataFrame, output_path: str):
    """
    保存alpha101数据到CSV
    
    Args:
        df: 因子DataFrame
        output_path: 输出文件路径
    """
    print(f"\n正在保存到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为CSV
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"保存成功！")
    print(f"  文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")
    print(f"  因子数量: {len([c for c in df.columns if 'alpha' in c])}")


def main():
    """主函数"""
    print("=" * 80)
    print("Alpha101 因子生成工具")
    print("=" * 80)
    
    try:
        # 1. 加载数据
        raw_data = load_hs300_data(DATA_PATH)
        
        # 2. 准备数据格式
        pivot_data = prepare_data_for_alphas(raw_data)
        
        # 3. 计算因子
        alpha_df = calculate_alpha_factors(pivot_data)
        
        # 4. 保存结果
        save_alpha101(alpha_df, OUTPUT_PATH)
        
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        
        # 显示数据预览
        print("\n数据预览（前5行）:")
        print(alpha_df.head())
        
        print("\n数据预览（因子统计）:")
        alpha_cols = [c for c in alpha_df.columns if 'alpha' in c]
        print(alpha_df[alpha_cols].describe())
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
