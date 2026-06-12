# import pandas as pd

# # 读取两个CSV文件（第一列是项目名，没有列标题）
# old_df = pd.read_csv('Old_overhead.csv', index_col=0)  # 将第一列作为行索引
# new_df = pd.read_csv('New_overhead.csv', index_col=0)

# # 重置索引，使项目名成为普通列
# old_df.reset_index(inplace=True)
# new_df.reset_index(inplace=True)

# # 重命名列以便区分
# old_df.columns = ['project', 'time_old', 'memory_old']
# new_df.columns = ['project', 'time_new', 'memory_new']

# # 按项目名合并
# merged = pd.merge(old_df, new_df, on='project', how='inner')

# # 计算时间差和内存差（新 - 旧）
# merged['time_diff'] = merged['time_new'] - merged['time_old']
# merged['memory_diff'] = merged['memory_new'] - merged['memory_old']

# # 计算百分比变化（注意避免除以0，但这里所有值都大于0）
# merged['time_pct'] = (merged['time_diff'] / merged['time_old']) * 100
# merged['memory_pct'] = (merged['memory_diff'] / merged['memory_old']) * 100

# # 生成格式化的对比字符串
# def format_change(old, new, diff, pct):
#     sign = '+' if diff >= 0 else ''
#     return f"{old:.2f} -> {new:.2f} [{sign}{diff:.2f} ({sign}{pct:.1f}%)]"

# merged['time_comparison'] = merged.apply(
#     lambda r: format_change(r['time_old'], r['time_new'], r['time_diff'], r['time_pct']), axis=1
# )
# merged['memory_comparison'] = merged.apply(
#     lambda r: format_change(r['memory_old'], r['memory_new'], r['memory_diff'], r['memory_pct']), axis=1
# )

# # 选择最终输出的列
# output = merged[[
#     'project',
#     'time_old', 'time_new', 'time_diff', 'time_pct', 'time_comparison',
#     'memory_old', 'memory_new', 'memory_diff', 'memory_pct', 'memory_comparison'
# ]]

# # 保存到新CSV文件
# output.to_csv('overhead_comparison.csv', index=False)

# print("对比结果已保存到 overhead_comparison.csv")

# 保留两位小数版本

import pandas as pd

# 读取两个CSV文件
old_df = pd.read_csv('Old_overhead.csv', index_col=0)
new_df = pd.read_csv('New_overhead.csv', index_col=0)

# 重置索引，使项目名成为普通列
old_df.reset_index(inplace=True)
new_df.reset_index(inplace=True)

# 重命名列
old_df.columns = ['project', 'time_old', 'memory_old']
new_df.columns = ['project', 'time_new', 'memory_new']

# 合并
merged = pd.merge(old_df, new_df, on='project', how='inner')

# 计算差值（保留原始精度，最后统一舍入）
merged['time_diff'] = merged['time_new'] - merged['time_old']
merged['memory_diff'] = merged['memory_new'] - merged['memory_old']
merged['time_pct'] = (merged['time_diff'] / merged['time_old']) * 100
merged['memory_pct'] = (merged['memory_diff'] / merged['memory_old']) * 100

# 生成保留两位小数的格式化对比字符串
def format_change_rounded(old, new, diff, pct):
    # 四舍五入到两位小数
    old_r = round(old, 2)
    new_r = round(new, 2)
    diff_r = round(diff, 2)
    pct_r = round(pct, 2)
    sign = '+' if diff_r >= 0 else ''
    return f"{old_r:.2f} -> {new_r:.2f} [{sign}{diff_r:.2f} ({sign}{pct_r:.2f}%)]"

merged['time_comparison'] = merged.apply(
    lambda r: format_change_rounded(r['time_old'], r['time_new'], r['time_diff'], r['time_pct']), axis=1
)
merged['memory_comparison'] = merged.apply(
    lambda r: format_change_rounded(r['memory_old'], r['memory_new'], r['memory_diff'], r['memory_pct']), axis=1
)

# 对所有数值列进行四舍五入（保留两位小数）
numeric_cols = ['time_old', 'time_new', 'time_diff', 'time_pct', 
                'memory_old', 'memory_new', 'memory_diff', 'memory_pct']
merged[numeric_cols] = merged[numeric_cols].round(2)

# 选择最终输出的列
output = merged[[
    'project',
    'time_old', 'time_new', 'time_diff', 'time_pct', 'time_comparison',
    'memory_old', 'memory_new', 'memory_diff', 'memory_pct', 'memory_comparison'
]]

# 保存到新CSV文件
output.to_csv('overhead_comparison_rounded.csv', index=False)

print("保留两位小数的对比结果已保存到 overhead_comparison_rounded.csv")