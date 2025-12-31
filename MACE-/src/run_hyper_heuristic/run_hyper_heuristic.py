from openai import OpenAI
import os
import importlib
from datetime import datetime
from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import search_file
import json

def run_hyper_heuristic(
    problem,
    heuristic,
    llm_config_file,
    heuristic_dir,
    test_data,
    iterations_scale_factor,
    result_dir,
    time_limit=None  # 新增：时间限制（秒）
):
    
    list_results = []
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    heuristic_name = heuristic.split(os.sep)[-1].split(".")[0]
    
    heuristic_pool_path = os.path.join("src", "problems", problem, "heuristics", heuristic_dir)
    if not os.path.exists(heuristic_pool_path):
        raise FileNotFoundError(f"启发式目录不存在: {heuristic_pool_path}")
    heuristic_pool = [
        f for f in os.listdir(heuristic_pool_path) 
        if f != '.ipynb_checkpoints' 
    ]

    base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "output") if os.getenv("AMLT_OUTPUT_DIR") else "output"

  
    hyper_heuristic = None
    experiment_name = ""
    llm_client = None

    experiment_name = heuristic_name
    hyper_heuristic = SingleHyperHeuristic(heuristic=heuristic_name, problem=problem)

    try:
        module = importlib.import_module(f"src.problems.{problem}.env")
        Env = getattr(module, "Env")
    except Exception as e:
        raise ImportError(f"无法导入问题 {problem} 的环境类: {str(e)}")

    # 处理测试数据：如果test_data为"test_data"，则获取该目录下的所有文件；否则使用指定的单个文件
    # 处理测试数据（新增过滤逻辑）
    if test_data == "test_data":
        test_data_dir = search_file("test_data", problem)
        # 过滤掉 .ipynb_checkpoints 文件夹
        test_data_list = [
            f for f in os.listdir(test_data_dir) 
            if f != ".ipynb_checkpoints"  # 排除Jupyter自动生成的检查点文件夹
        ]
    elif test_data == "smoke_data":
        test_data_dir = search_file("smoke_data", problem)
        # 过滤掉 .ipynb_checkpoints 文件夹
        test_data_list = [
            f for f in os.listdir(test_data_dir) 
            if f != ".ipynb_checkpoints"  # 排除Jupyter自动生成的检查点文件夹
        ]

    else:
        test_data_list = [test_data]
        
    for data_name in test_data_list:
        try:
            # 初始化环境实例，设置输出目录并重置环境
            env = Env(data_name=data_name)
            output_dir = os.path.join(base_output_dir, problem, result_dir, env.data_ref_name, experiment_name)
            env.reset(output_dir)

            # 保存参数到parameters.txt文件
            params = {
                "problem": problem,
                "heuristic": heuristic,
                "llm_config_file": llm_config_file,
                "heuristic_dir": heuristic_dir,
                "test_data": test_data,
                "iterations_scale_factor": iterations_scale_factor,
                "result_dir": result_dir,
                "data_path": env.data_path,  # 存储具体问题数据的文件所在的位置
                "time_limit": time_limit  # 新增：保存时间限制参数
            }
            with open(os.path.join(env.output_dir, "parameters.txt"), 'w') as f:
                f.write('\n'.join(f'{k}={v}' for k, v in params.items()))

            # 运行启发式算法，传入时间限制参数
            validation_result = hyper_heuristic.run(env, time_limit=time_limit)

            # 输出结果：如果验证成功，保存结果并打印成功信息；否则打印失败信息
            if validation_result:
                env.dump_result()
                list_results.append(env.key_value)
            else:
                
                penalty_value =1e10
                list_results.append(penalty_value)
        # 捕获并打印处理当前测试数据时的错误
        except Exception as e:
            pass
    return list_results

def evaluate_all_heuristics(
    problem,
    heuristic_dir,
    test_data,
    llm_config_file=None,
    iterations_scale_factor=1.0,
    result_dir="result",
    save_to_file=True,
    time_limit=None # 新增：时间限制（秒）
):
    # 获取所有启发式文件

    heuristic_pool_path = os.path.join("src", "problems", problem, "heuristics", heuristic_dir)
    if not os.path.exists(heuristic_pool_path):
        raise FileNotFoundError(f"启发式目录不存在: {heuristic_pool_path}")
    
    # 获取所有启发式（去除扩展名.py）
    all_heuristics = [
        f.replace('.py', '') for f in os.listdir(heuristic_pool_path) 
        if f.endswith('.py') and f != '__init__.py' and not f.startswith('.')
    ]
    print(f"找到 {len(all_heuristics)} 个启发式算法:")
    
    # 存储所有结果
    results_dict = {}
    
    for i, heuristic_name in enumerate(all_heuristics, 1):
     
        try:
            # 调用原有的run_hyper_heuristic函数
            list_results = run_hyper_heuristic(
                problem=problem,
                heuristic=heuristic_name,
                llm_config_file=llm_config_file,
                heuristic_dir=heuristic_dir,
                test_data=test_data,
                iterations_scale_factor=iterations_scale_factor,
                result_dir=result_dir,
                time_limit=time_limit  # 新增：传递时间限制参数
            )
            
            # 保存结果
            results_dict[heuristic_name] = list_results
                    
        except Exception as e:
            results_dict[heuristic_name] = None
    
    # 打印汇总结果
    print(f"\n{'='*60}")
    print("test_data评估完成！结果汇总:")
    print(f"{'='*60}")
    
    return results_dict