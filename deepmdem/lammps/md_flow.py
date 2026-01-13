"""
MD工作流定义模块

本模块定义了完整的分子动力学模拟工作流，包括结构弛豫、加热、稳定化和平衡态模拟四个阶段。
使用dflow工作流管理系统实现各个步骤的自动化执行和数据传递。
"""
# 导入MD操作类
from deepmdem.lammps import MDRelax, MDHeating, MDStabilization, MDEquilibrium
# 导入dflow工作流相关类
from dflow import Step, Workflow, upload_artifact
from dflow.python import PythonOPTemplate
from pathlib import Path  # 文件路径处理
from dflow import SlurmRemoteExecutor  # Slurm远程执行器
from dflow import Workflow  # 工作流类


def md_workflow(
    temp: float,
    element_map: dict,
    running_cores: int,
    nodes: int,
    add_vacuum: float,
    potential: Path,
    input_structure: Path,
    excutor: SlurmRemoteExecutor = None,
) -> Workflow:
    """
    创建并提交MD工作流
    
    工作流包含四个连续步骤：
    1. Relax: 结构弛豫，优化原子位置和晶格参数
    2. Heating: 从低温加热到目标温度
    3. Stabilization: 在目标温度下稳定系统
    4. Equilibrium: 平衡态MD模拟，收集声子配置
    
    参数:
        temp: 目标温度（K）
        element_map: 元素类型映射字典，如 {'Al': 0}
        running_cores: 每个步骤使用的CPU核心数
        nodes: 使用的计算节点数
        add_vacuum: 添加真空层的厚度（Å），用于表面体系
        potential: 势能文件路径
        input_structure: 输入结构文件路径
        excutor: Slurm远程执行器
    
    返回:
        Workflow: 已提交的工作流对象
    """
    # 上传势能文件和结构文件到工作流系统
    potential = upload_artifact(potential)
    input_structure = upload_artifact(input_structure)

    # 步骤1: 结构弛豫（Relax）
    # 目的：优化原子位置和晶格参数，使系统达到能量最低状态
    step0 = Step(
        name="Relax",  # 步骤名称
        template=PythonOPTemplate(MDRelax, command=["source ~/dflow.env && python"]),  # 使用MDRelax操作
        parameters={
            "pbc": True,  # 使用周期性边界条件
            "element_map": element_map,  # 元素映射
            "potential_type": "deepmd",  # 使用深度势能
            "running_cores": running_cores,  # 并行核心数
        },
        artifacts={"atomic_potential": potential, "structure": input_structure},  # 输入文件
        executor=excutor(nodes),  # 执行器配置
    )

    # 步骤2: 加热（Heating）
    # 目的：将系统从低温（10K）加热到目标温度
    # 使用NPT系综，允许晶格参数随温度变化
    step1 = Step(
        name="Heating",  # 步骤名称
        template=PythonOPTemplate(MDHeating, command=["source ~/dflow.env && python"]),  # 使用MDHeating操作
        parameters={
            "temp": temp,  # 目标温度
            "nsteps": 3000,  # 模拟步数
            "ensemble": "npt",  # 使用NPT系综（恒定粒子数、压力、温度）
            "pres": 1,  # 压力（bar）
            "pbc": True,  # 使用周期性边界条件
            "timestep": 0.005,  # 时间步长（ps）
            "neidelay": 10,  # 邻居列表更新延迟
            "trj_freq": 100,  # 轨迹输出频率
            "element_map": element_map,  # 元素映射
            "potential_type": "deepmd",  # 使用深度势能
            "running_cores": running_cores,  # 并行核心数
        },
        artifacts={
            "atomic_potential": potential,  # 势能文件
            "structure": step0.outputs.artifacts["out_structure"],  # 使用上一步弛豫后的结构
        },
        executor=excutor(nodes),  # 执行器配置
    )

    # 步骤3: 稳定化（Stabilization）
    # 目的：在目标温度下稳定系统，包括结构优化和NVT模拟
    # 特点：可以添加真空层，用于模拟表面体系
    step2 = Step(
        name="Stabilization",  # 步骤名称
        template=PythonOPTemplate(
            MDStabilization, command=["source ~/dflow.env && python"]  # 使用MDStabilization操作
        ),
        parameters={
            "temp": temp,  # 目标温度
            "nsteps": 1000,  # 模拟步数
            "pbc": True,  # 使用周期性边界条件
            "timestep": 0.005,  # 时间步长（ps）
            "neidelay": 10,  # 邻居列表更新延迟
            "trj_freq": 100,  # 轨迹输出频率
            "pres": 1,  # 压力（bar）
            "element_map": element_map,  # 元素映射
            "potential_type": "deepmd",  # 使用深度势能
            "add_vacuum": add_vacuum,  # 真空层厚度（Å）
            "running_cores": running_cores,  # 并行核心数
        },
        artifacts={
            "atomic_potential": potential,  # 势能文件
            "structure": step1.outputs.artifacts["out_structure"],  # 使用上一步加热后的结构
        },
        executor=excutor(nodes),  # 执行器配置
    )

    # 步骤4: 平衡态模拟（Equilibrium）
    # 目的：在目标温度下进行长时间的NVT模拟，收集声子配置
    # 这些配置将用于后续的电子散射模拟
    step3 = Step(
        name="Equilibrium",  # 步骤名称
        template=PythonOPTemplate(
            MDEquilibrium, command=["source ~/dflow.env && python"]  # 使用MDEquilibrium操作
        ),
        parameters={
            "temp": temp,  # 目标温度
            "nsteps": 10000,  # 模拟步数（比前几步多，以收集足够的声子配置）
            "pbc": True,  # 使用周期性边界条件
            "timestep": 0.005,  # 时间步长（ps）
            "neidelay": 10,  # 邻居列表更新延迟
            "trj_freq": 10,  # 轨迹输出频率（更频繁，以获取更多声子配置）
            "element_map": element_map,  # 元素映射
            "potential_type": "deepmd",  # 使用深度势能
            "running_cores": running_cores,  # 并行核心数
        },
        artifacts={
            "atomic_potential": potential,  # 势能文件
            "structure": step2.outputs.artifacts["out_structure"],  # 使用上一步稳定化后的结构
        },
        executor=excutor(nodes),  # 执行器配置
    )
    
    # 创建工作流并添加所有步骤
    wf = Workflow(name=f"md-{temp}k")  # 创建工作流，名称包含温度信息
    wf.add(step0)  # 添加结构弛豫步骤
    wf.add(step1)  # 添加加热步骤
    wf.add(step2)  # 添加稳定化步骤
    wf.add(step3)  # 添加平衡态模拟步骤
    wf.submit()  # 提交工作流执行

    return wf  # 返回工作流对象
