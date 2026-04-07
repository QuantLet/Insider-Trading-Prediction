import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import font_manager


def init_project_environment():
    """
    初始化项目环境：生成时间戳、创建目录、初始化 Markdown 日志、设置中文字体（如有）。

    返回：
        env_dict: 包含当前时间、路径配置、起始时间等信息的字典
    """
    # 当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 路径配置
    project_root = os.path.abspath(".")
    data_dir = os.path.join(project_root, "A_Data")
    log_dir = os.path.join(project_root, "B_log")
    result_dir = os.path.join(project_root, "D_Result")
    model_dir = os.path.join(project_root, "C_Model")
    font_path = os.path.join(project_root, "fonts", "NotoSerifCJKsc-Regular.otf")

    # 创建目录
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Markdown 日志路径
    md_log_path = os.path.join(log_dir, f"log_{current_time}.md")
    with open(md_log_path, 'w', encoding='utf-8') as f:
        f.write(f"# 实验日志 {current_time}\n\n")

    # 设置中文字体（注册 + 应用）
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_prop = font_manager.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"成功启用中文字体: {font_name}")
    else:
        print("未找到中文字体文件，中文内容可能显示异常。")

    # 起始时间
    start_time = time.time()

    # 控制台日志输出
    print("当前时间:", current_time)
    print("根目录路径:", project_root)
    print("数据目录路径data_dir:", data_dir)
    print("结果保存路径result_dir:", result_dir)
    print("模型保存路径model_dir:", model_dir)
    print("Markdown日志路径md_log:", md_log_path)

    return {
        'current_time': current_time,
        'project_root': project_root,
        'data_dir': data_dir,
        'result_dir': result_dir,
        'model_dir': model_dir,
        'log_dir': log_dir,
        'md_log': md_log_path,
        'start_time': start_time
    }


