"""
GUI数据采集程序启动脚本
提供图形界面的一键数据获取功能
"""

import sys
import os
from pathlib import Path

def main():
    """主函数"""
    try:
        # 添加项目根目录到Python路径
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        # 添加data_pipeline目录到Python路径
        sys.path.insert(0, str(Path(__file__).parent))
        
        print("正在启动GUI数据采集程序...")
        print(f"项目根目录: {project_root}")
        
        # 检查依赖包
        try:
            import tkinter
            import pandas
            import akshare as ak
            print("✓ 所有依赖包已正确安装")
        except ImportError as e:
            print(f"✗ 依赖包缺失: {e}")
            print("请运行: pip install -r requirements.txt")
            input("按回车键退出...")
            return
            
        from gui_data_collector import main as gui_main
        gui_main()
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请检查项目结构是否正确")
        input("按回车键退出...")
    except Exception as e:
        print(f"启动过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")

if __name__ == "__main__":
    main()