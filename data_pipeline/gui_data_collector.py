"""
GUI数据采集程序
提供图形界面的一键数据获取功能，包含所有可配置项
支持沪深300历史数据采集、断点续传、实时进度显示
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import os
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime, timedelta

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from data_collector.daily_price import DailyPriceCollector
    from data_collector.request_controller import RequestConfig
except ImportError as e:
    print(f"导入模块错误: {e}")
    # 提供友好的错误提示
    DailyPriceCollector = None
    RequestConfig = None

class DataCollectorGUI:
    """数据采集GUI应用程序"""
    
    def __init__(self, root):
        """
        初始化GUI界面
        
        Args:
            root: Tkinter根窗口
        """
        self.root = root
        self.root.title("沪深300历史数据采集器 v2.0")
        self.root.geometry("900x750")
        self.root.resizable(True, True)
        
        # 设置窗口图标（如果有的话）
        try:
            self.root.iconbitmap("")
        except:
            pass
        
        # 数据采集器实例
        self.collector = None
        
        # 采集线程
        self.collect_thread = None
        self.is_running = False
        
        # 日志处理器
        self.log_handler = None
        
        # 当前进度信息
        self.current_progress = 0
        self.total_progress = 100
        
        # 检查模块是否正常导入
        if DailyPriceCollector is None or RequestConfig is None:
            messagebox.showerror("初始化错误", "无法导入必要的模块，请检查项目结构")
            return
        
        self.setup_ui()
        self.setup_logging()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 数据采集配置区域
        config_frame = ttk.LabelFrame(main_frame, text="数据采集配置", padding="10")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(3, weight=1)
        
        # 第一行：年份范围配置
        ttk.Label(config_frame, text="开始年份:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.start_year_var = tk.StringVar(value="2019")
        start_year_entry = ttk.Entry(config_frame, textvariable=self.start_year_var, width=10)
        start_year_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(config_frame, text="结束年份:").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.end_year_var = tk.StringVar(value="2024")
        end_year_entry = ttk.Entry(config_frame, textvariable=self.end_year_var, width=10)
        end_year_entry.grid(row=0, column=3, sticky=tk.W, pady=5)
        
        # 第二行：请求控制配置
        ttk.Label(config_frame, text="基础延迟(秒):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.base_delay_var = tk.StringVar(value="2.0")
        ttk.Entry(config_frame, textvariable=self.base_delay_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(config_frame, text="最大延迟(秒):").grid(row=1, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.max_delay_var = tk.StringVar(value="30.0")
        ttk.Entry(config_frame, textvariable=self.max_delay_var, width=10).grid(row=1, column=3, sticky=tk.W, pady=5)
        
        # 第三行：重试和批次配置
        ttk.Label(config_frame, text="最大重试次数:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_retries_var = tk.StringVar(value="5")
        ttk.Entry(config_frame, textvariable=self.max_retries_var, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(config_frame, text="批次大小:").grid(row=2, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.batch_size_var = tk.StringVar(value="30")
        ttk.Entry(config_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=3, sticky=tk.W, pady=5)
        
        # 第四行：请求频率配置
        ttk.Label(config_frame, text="每分钟请求数:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.requests_per_minute_var = tk.StringVar(value="20")
        ttk.Entry(config_frame, textvariable=self.requests_per_minute_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(config_frame, text="数据接口:").grid(row=3, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.api_var = tk.StringVar(value="auto")
        api_frame = ttk.Frame(config_frame)
        api_frame.grid(row=3, column=3, sticky=tk.W, pady=5)
        ttk.Radiobutton(api_frame, text="自动选择", variable=self.api_var, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(api_frame, text="主接口", variable=self.api_var, value="primary").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Radiobutton(api_frame, text="备用接口", variable=self.api_var, value="backup").pack(side=tk.LEFT, padx=(10, 0))
        
        # 第五行：采集模式选择
        ttk.Label(config_frame, text="采集模式:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="resume")
        mode_frame = ttk.Frame(config_frame)
        mode_frame.grid(row=4, column=1, columnspan=3, sticky=tk.W, pady=5)
        ttk.Radiobutton(mode_frame, text="断点续传", variable=self.mode_var, value="resume").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="重新采集", variable=self.mode_var, value="restart").pack(side=tk.LEFT, padx=(20, 0))
        
        # 第六行：数据验证选项
        ttk.Label(config_frame, text="数据验证:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.validation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="启用数据完整性检查", variable=self.validation_var).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(config_frame, text="保存格式:").grid(row=5, column=2, sticky=tk.W, pady=5, padx=(20, 0))
        self.format_var = tk.StringVar(value="parquet")
        format_frame = ttk.Frame(config_frame)
        format_frame.grid(row=5, column=3, sticky=tk.W, pady=5)
        ttk.Radiobutton(format_frame, text="Parquet", variable=self.format_var, value="parquet").pack(side=tk.LEFT)
        ttk.Radiobutton(format_frame, text="CSV", variable=self.format_var, value="csv").pack(side=tk.LEFT, padx=(10, 0))
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # 第一行按钮
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(button_row1, text="🚀 开始采集", command=self.start_collection, width=12)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_row1, text="⏹️ 停止采集", command=self.stop_collection, state=tk.DISABLED, width=12)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.check_status_button = ttk.Button(button_row1, text="📊 检查状态", command=self.check_data_status, width=12)
        self.check_status_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_button = ttk.Button(button_row1, text="📁 导出数据", command=self.export_data, width=12)
        self.export_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 第二行按钮
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(fill=tk.X, pady=5)
        
        self.test_api_button = ttk.Button(button_row2, text="🔧 测试接口", command=self.test_api_connection, width=12)
        self.test_api_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_log_button = ttk.Button(button_row2, text="🗑️ 清空日志", command=self.clear_log, width=12)
        self.clear_log_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.settings_button = ttk.Button(button_row2, text="⚙️ 高级设置", command=self.show_settings, width=12)
        self.settings_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.help_button = ttk.Button(button_row2, text="❓ 帮助", command=self.show_help, width=12)
        self.help_button.pack(side=tk.LEFT)
        
        # 进度显示
        self.progress_var = tk.StringVar(value="准备就绪")
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate', maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # 进度百分比显示
        self.progress_percent_var = tk.StringVar(value="0%")
        self.progress_percent_label = ttk.Label(main_frame, textvariable=self.progress_percent_var)
        self.progress_percent_label.grid(row=3, column=1, sticky=tk.E, pady=5, padx=(0, 10))
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(main_frame, text="采集日志", padding="10")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 状态信息区域
        status_frame = ttk.LabelFrame(main_frame, text="状态信息", padding="10")
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="等待开始采集...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 配置主框架的行权重
        main_frame.rowconfigure(4, weight=1)
        
    def setup_logging(self):
        """设置日志系统"""
        # 创建自定义日志处理器
        self.log_handler = TextHandler(self.log_text)
        self.log_handler.setLevel(logging.INFO)
        
        # 配置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        
        # 获取数据采集器的日志器并添加处理器
        collector_logger = logging.getLogger('data_collector.daily_price')
        collector_logger.setLevel(logging.INFO)
        collector_logger.addHandler(self.log_handler)
        
        # 添加请求控制器的日志器
        controller_logger = logging.getLogger('data_collector.request_controller')
        controller_logger.setLevel(logging.INFO)
        controller_logger.addHandler(self.log_handler)
        
    def start_collection(self):
        """开始数据采集"""
        if self.is_running:
            messagebox.showwarning("警告", "采集任务正在运行中")
            return
            
        # 验证输入参数
        try:
            start_year = int(self.start_year_var.get())
            end_year = int(self.end_year_var.get())
            base_delay = float(self.base_delay_var.get())
            max_delay = float(self.max_delay_var.get())
            max_retries = int(self.max_retries_var.get())
            batch_size = int(self.batch_size_var.get())
            requests_per_minute = int(self.requests_per_minute_var.get())
            
            if start_year > end_year:
                messagebox.showerror("错误", "开始年份不能大于结束年份")
                return
                
        except ValueError as e:
            messagebox.showerror("错误", f"参数格式错误: {e}")
            return
        
        # 更新UI状态
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.progress_percent_var.set("0%")
        self.progress_var.set("正在采集数据...")
        
        # 清空日志
        self.log_text.delete(1.0, tk.END)
        
        # 在新线程中运行采集任务
        self.collect_thread = threading.Thread(target=self.run_collection, daemon=True)
        self.collect_thread.start()
        
    def stop_collection(self):
        """停止数据采集"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.progress_var.set("正在停止采集...")
        
        # 设置停止标志，让采集器优雅退出
        if self.collector:
            # 在采集器中添加停止标志
            self.collector.stop_requested = True
            
        self.log_info("用户请求停止采集，正在优雅退出...")
        
        # 强制停止线程（如果优雅退出失败）
        def force_stop():
            if self.is_running:
                self.log_warning("采集线程未正常退出，强制停止")
                self.is_running = False
                self.collection_finished()
        
        # 5秒后强制停止
        self.root.after(5000, force_stop)
        
    def run_collection(self):
        """运行数据采集任务"""
        try:
            # 创建请求配置
            request_config = RequestConfig(
                base_delay=float(self.base_delay_var.get()),
                max_delay=float(self.max_delay_var.get()),
                max_retries=int(self.max_retries_var.get()),
                backoff_factor=2.0,
                jitter=0.2,
                batch_size=int(self.batch_size_var.get()),
                requests_per_minute=int(self.requests_per_minute_var.get())
            )
            
            # 创建数据采集器，并传入配置
            self.collector = DailyPriceCollector()
            # 重新初始化请求控制器，使用新的配置
            from data_collector.request_controller import RequestController
            self.collector.request_controller = RequestController(request_config)
            
            # 根据模式选择采集方法
            start_year = int(self.start_year_var.get())
            end_year = int(self.end_year_var.get())
            
            if self.mode_var.get() == "resume":
                self.update_status("开始断点续传采集...")
                self.collector.collect_hs300_historical_data_resume(start_year, end_year, self.update_progress)
            else:
                self.update_status("开始重新采集数据...")
                self.collector.collect_hs300_historical_data(start_year, end_year, self.update_progress)
                
            # 采集完成
            self.update_status("数据采集完成")
            
        except Exception as e:
            error_msg = f"采集过程中出现错误: {e}"
            self.log_error(error_msg)
            self.update_status("采集失败")
            
        finally:
            # 恢复UI状态
            self.root.after(0, self.collection_finished)
    
    def update_progress(self, current, total):
        """更新进度条显示
        
        Args:
            current: 当前进度
            total: 总进度
        """
        def update():
            if total > 0:
                progress_percent = int((current / total) * 100)
                self.progress_bar['value'] = progress_percent
                self.progress_percent_var.set(f"{progress_percent}%")
                self.progress_var.set(f"进度: {current}/{total} 批次 ({progress_percent}%)")
            else:
                self.progress_bar['value'] = 0
                self.progress_percent_var.set("0%")
                self.progress_var.set("进度: 计算中...")
        
        # 在主线程中更新UI
        self.root.after(0, update)
            
    def collection_finished(self):
        """采集完成后的UI更新"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar['value'] = 100
        self.progress_percent_var.set("100%")
        self.progress_var.set("采集完成")
        
    def check_data_status(self):
        """检查当前数据状态"""
        try:
            collector = DailyPriceCollector()
            
            # 检查数据文件
            existing_files = list(collector.price_dir.glob("daily_prices_*.parquet"))
            if not existing_files:
                self.update_status("未找到数据文件")
                return
                
            # 读取最新文件
            latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            
            # 获取统计信息
            total_records = len(df)
            unique_stocks = df['股票代码'].nunique() if '股票代码' in df.columns else 0
            date_range = f"{df['日期'].min()} 至 {df['日期'].max()}" if '日期' in df.columns else "未知"
            
            status_info = f"数据状态: {total_records} 条记录, {unique_stocks} 只股票, 时间范围: {date_range}"
            self.update_status(status_info)
            
            # 在日志中显示详细信息
            self.log_info(f"数据文件: {latest_file.name}")
            self.log_info(f"总记录数: {total_records}")
            self.log_info(f"股票数量: {unique_stocks}")
            self.log_info(f"时间范围: {date_range}")
            
        except Exception as e:
            error_msg = f"检查数据状态时出错: {e}"
            self.log_error(error_msg)
            self.update_status("检查状态失败")
            
    def export_data(self):
        """导出数据到CSV格式"""
        try:
            from tkinter import filedialog
            
            collector = DailyPriceCollector()
            existing_files = list(collector.price_dir.glob("daily_prices_*.parquet"))
            if not existing_files:
                messagebox.showwarning("警告", "未找到数据文件")
                return
                
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
                title="导出数据"
            )
            
            if file_path:
                latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_parquet(latest_file)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
                self.log_info(f"数据已导出到: {file_path}")
                messagebox.showinfo("成功", f"数据已成功导出到:\n{file_path}")
                
        except Exception as e:
            error_msg = f"导出数据时出错: {e}"
            self.log_error(error_msg)
            messagebox.showerror("错误", error_msg)
            
    def test_api_connection(self):
        """测试API接口连接"""
        try:
            import akshare as ak
            
            self.log_info("正在测试AKShare接口连接...")
            
            # 测试主接口
            test_data = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240101", end_date="20240110", adjust="qfq")
            if test_data is not None and len(test_data) > 0:
                self.log_info("✓ 主接口(stock_zh_a_hist)连接正常")
            else:
                self.log_warning("⚠ 主接口返回空数据")
                
            # 测试备用接口（注意：stock_zh_a_daily接口不支持start_date和end_date参数）
            test_data2 = ak.stock_zh_a_daily(symbol="000001", adjust="qfq")
            if test_data2 is not None and len(test_data2) > 0:
                self.log_info("✓ 备用接口(stock_zh_a_daily)连接正常")
            else:
                self.log_warning("⚠ 备用接口返回空数据")
                
            self.log_info("接口测试完成")
            
        except Exception as e:
            error_msg = f"接口测试失败: {e}"
            self.log_error(error_msg)
            
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
        self.log_info("日志已清空")
        
    def show_settings(self):
        """显示高级设置对话框"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("高级设置")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # 设置窗口内容
        ttk.Label(settings_window, text="高级设置", font=("Arial", 12, "bold")).pack(pady=10)
        
        # 添加一些高级设置选项
        ttk.Label(settings_window, text="数据保存路径:").pack(anchor=tk.W, padx=20, pady=5)
        path_frame = ttk.Frame(settings_window)
        path_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.data_path_var = tk.StringVar(value=str(Path(__file__).parent / "data" / "daily_prices"))
        ttk.Entry(path_frame, textvariable=self.data_path_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="浏览", command=self.browse_data_path).pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Label(settings_window, text="日志级别:").pack(anchor=tk.W, padx=20, pady=5)
        self.log_level_var = tk.StringVar(value="INFO")
        log_frame = ttk.Frame(settings_window)
        log_frame.pack(fill=tk.X, padx=20, pady=5)
        
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for level in levels:
            ttk.Radiobutton(log_frame, text=level, variable=self.log_level_var, value=level).pack(side=tk.LEFT)
            
        ttk.Button(settings_window, text="保存设置", command=self.save_settings).pack(pady=20)
        
    def browse_data_path(self):
        """浏览数据保存路径"""
        from tkinter import filedialog
        path = filedialog.askdirectory(title="选择数据保存路径")
        if path:
            self.data_path_var.set(path)
            
    def save_settings(self):
        """保存高级设置"""
        # 这里可以添加保存设置的逻辑
        messagebox.showinfo("提示", "设置已保存（演示功能）")
        
    def show_help(self):
        """显示帮助信息"""
        help_text = """
沪深300历史数据采集器 v2.0

功能说明：
1. 支持沪深300成分股历史数据采集
2. 支持断点续传和重新采集两种模式
3. 自动选择最优数据接口
4. 实时进度显示和日志记录

使用说明：
1. 设置采集年份范围
2. 配置请求参数（延迟、重试等）
3. 选择采集模式
4. 点击开始采集

技术支持：如有问题请联系开发团队
        """
        messagebox.showinfo("帮助", help_text.strip())
            
    def update_status(self, message):
        """更新状态信息"""
        def update():
            self.status_var.set(message)
        self.root.after(0, update)
        
    def log_info(self, message):
        """记录信息日志"""
        def log():
            self.log_text.insert(tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {message}\n")
            self.log_text.see(tk.END)
        self.root.after(0, log)
        
    def log_error(self, message):
        """记录错误日志"""
        def log():
            self.log_text.insert(tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {message}\n")
            self.log_text.see(tk.END)
        self.root.after(0, log)
        
    def log_warning(self, message):
        """记录警告日志"""
        def log():
            self.log_text.insert(tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {message}\n")
            self.log_text.see(tk.END)
        self.root.after(0, log)


class TextHandler(logging.Handler):
    """自定义日志处理器，将日志输出到Text组件"""
    
    def __init__(self, text_widget):
        """
        初始化文本处理器
        
        Args:
            text_widget: Tkinter Text组件
        """
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        """处理日志记录"""
        msg = self.format(record)
        
        def append():
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            
        # 在主线程中更新UI
        self.text_widget.after(0, append)


def main():
    """主函数"""
    root = tk.Tk()
    app = DataCollectorGUI(root)
    
    # 启动时自动检查数据状态
    root.after(100, app.check_data_status)
    
    root.mainloop()


if __name__ == "__main__":
    main()