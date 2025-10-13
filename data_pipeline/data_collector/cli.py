"""
数据采集命令行工具
提供命令行接口来运行数据采集程序
"""

import argparse
import sys
import os
from datetime import datetime

from data_collector import HS300DataCollector
import config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='沪深300成分股日频数据采集工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python cli.py                            # 使用默认配置运行
  python cli.py --start 2020-01-01        # 指定开始日期
  python cli.py --batch-size 10           # 指定批次大小
  python cli.py --resume                   # 恢复模式，跳过已处理的股票
        """
    )
    
    parser.add_argument('--start-date', '-s', 
                       default=config.START_DATE,
                       help=f'数据开始日期 (默认: {config.START_DATE})')
    
    parser.add_argument('--end-date', '-e',
                       default=config.END_DATE,
                       help=f'数据结束日期 (默认: {config.END_DATE})')
    
    parser.add_argument('--batch-size', '-b',
                       type=int, default=config.BATCH_SIZE,
                       help=f'每批次处理的股票数量 (默认: {config.BATCH_SIZE})')
    
    parser.add_argument('--delay', '-d',
                       type=float, default=2.0,
                       help='请求间隔时间，秒 (默认: 2.0)')
    
    parser.add_argument('--max-retries', '-r',
                       type=int, default=3,
                       help='最大重试次数 (默认: 3)')
    
    parser.add_argument('--resume', action='store_true',
                       help='恢复模式，跳过已处理的股票')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式，不实际获取数据')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出模式')
    
    return parser.parse_args()


def validate_date(date_str):
    """验证日期格式"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def main():
    """主函数"""
    args = parse_arguments()
    
    # 验证日期格式
    if not validate_date(args.start_date):
        print(f"错误: 开始日期格式无效: {args.start_date}，请使用 YYYY-MM-DD 格式")
        sys.exit(1)
    
    if not validate_date(args.end_date):
        print(f"错误: 结束日期格式无效: {args.end_date}，请使用 YYYY-MM-DD 格式")
        sys.exit(1)
    
    # 检查日期范围
    if args.start_date > args.end_date:
        print(f"错误: 开始日期 {args.start_date} 不能晚于结束日期 {args.end_date}")
        sys.exit(1)
    
    # 显示配置信息
    print("=" * 60)
    print("沪深300成分股数据采集工具")
    print("=" * 60)
    print(f"时间范围: {args.start_date} 到 {args.end_date}")
    print(f"批次大小: {args.batch_size}")
    print(f"请求间隔: {args.delay} 秒")
    print(f"最大重试: {args.max_retries}")
    print(f"恢复模式: {'是' if args.resume else '否'}")
    print(f"试运行: {'是' if args.dry_run else '否'}")
    print("=" * 60)
    
    if args.dry_run:
        print("试运行模式: 不实际获取数据")
        print("配置验证完成，可以正常运行")
        return
    
    # 确认是否继续
    if not args.resume:
        response = input("是否开始数据采集? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("操作已取消")
            return
    
    try:
        # 创建并运行采集器
        collector = HS300DataCollector()
        
        # 更新配置参数
        config.START_DATE = args.start_date
        config.END_DATE = args.end_date
        config.BATCH_SIZE = args.batch_size
        config.REQUEST_DELAY = args.delay
        config.MAX_RETRIES = args.max_retries
        
        collector.run()
        
        print("数据采集完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()