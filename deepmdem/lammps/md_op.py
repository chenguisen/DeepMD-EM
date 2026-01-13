# 导入必要的模块
from pathlib import Path  # 用于处理文件路径
from os.path import basename  # 获取文件名
from dflow.python import OP, OPIO, Artifact, OPIOSign  # dflow工作流相关类
from typing import List  # 类型提示

# import random  # 暂未使用的随机数模块
import os  # 操作系统接口
import subprocess  # 用于执行系统命令
import dpdata  # 用于处理分子动力学数据
from deepmdem.lammps.periodic_table import elements_dict  # 元素周期表数据


class MDRelax(OP):
    """
    MD结构弛豫操作类
    
    功能：对输入的晶体结构进行能量最小化，优化原子位置和晶格参数
    """
    def __init__(self):
        """初始化方法，目前为空"""  
        pass

    @classmethod
    def get_input_sign(cls):
        """
        定义输入参数的类型和结构
        
        返回:
            OPIOSign对象，包含以下输入参数:
            - pbc: 是否使用周期性边界条件
            - atomic_potential: 原子势能文件路径
            - structure: 输入结构文件路径
            - element_map: 元素类型映射字典
            - potential_type: 势能类型（deepmd或eam）
            - running_cores: 并行计算使用的核心数
        """
        return OPIOSign(
            {
                "pbc": bool,  # 周期性边界条件
                "atomic_potential": Artifact(Path),  # 势能文件
                "structure": Artifact(Path),  # 结构文件
                "element_map": dict,  # 元素映射
                "potential_type": str,  # 势能类型
                "running_cores": int,  # 计算核心数
            }
        )

    @classmethod
    def get_output_sign(cls):
        """
        定义输出参数的类型和结构
        
        返回:
            OPIOSign对象，包含以下输出:
            - out_art: 输出文件列表（输入文件、日志文件、轨迹文件）
            - out_structure: 弛豫后的结构文件
        """
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),  # 输出文件列表
                "out_structure": Artifact(Path),  # 弛豫后的结构
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        """
        执行结构弛豫计算
        
        参数:
            op_in: 输入参数对象，包含结构、势能等信息
        
        返回:
            op_out: 输出参数对象，包含计算结果文件
        
        流程:
            1. 切换到势能文件所在目录
            2. 生成LAMMPS输入文件
            3. 运行LAMMPS进行结构优化
            4. 读取并转换输出结构
            5. 返回结果文件
        """
        # 保存当前工作目录
        cwd = os.getcwd()
        # 切换到势能文件所在目录
        os.chdir(op_in["atomic_potential"].parent)

        # 生成LAMMPS输入文件内容
        ret = self.make_lmp_input(
            conf_file=op_in["structure"],  # 输入结构文件
            atomic_potential=basename(op_in["atomic_potential"]),  # 势能文件名
            pbc=op_in["pbc"],  # 周期性边界条件
            element_map=op_in["element_map"],  # 元素映射
            potential_type=op_in["potential_type"],  # 势能类型
        )
        # 写入LAMMPS输入文件
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        # 构建并执行LAMMPS命令
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        # 读取LAMMPS输出文件
        system = dpdata.System(
            "relax.dump",  # LAMMPS轨迹文件
            fmt="lammps/dump",  # 文件格式
            type_map=list(op_in["element_map"].keys()),  # 元素类型映射
            unwrap=True,  # 解包周期性边界
        )
        # 将结构保存为LAMMPS格式，取最后一帧（优化后的结构）
        system.to("lammps/lmp", "output_structure.lammps", frame_idx=-1)
        # 准备输出
        op_out = OPIO(
            {
                "out_art": [Path("input.lammps"), Path("log"), Path("relax.dump")],  # 输出文件列表
                "out_structure": Path("output_structure.lammps"),  # 优化后的结构
            }
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        atomic_potential: str,
        potential_type: str,
        element_map: dict,
        neidelay: int = 1,
        trj_freq: int = 1,
        pbc: bool = True,
    ):
        """
        生成LAMMPS结构弛豫的输入文件内容
        
        参数:
            conf_file: 结构文件路径
            atomic_potential: 势能文件名
            potential_type: 势能类型（deepmd或eam）
            element_map: 元素类型映射字典
            neidelay: 邻居列表更新延迟步数
            trj_freq: 轨迹输出频率
            pbc: 是否使用周期性边界条件
        
        返回:
            str: LAMMPS输入文件的完整内容
        """
        # 设置热力学和轨迹输出频率变量
        ret = "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
        ret += "\n"
        # 设置单位制为金属单位（eV, Å, ps）
        ret += "units           metal\n"
        # 设置边界条件
        if pbc is False:
            ret += "boundary        f f f\n"  # 非周期性边界
        else:
            ret += "boundary        p p p\n"  # 周期性边界
        # 设置原子样式
        ret += "atom_style      atomic\n"
        ret += "\n"
        # 设置邻居列表参数
        ret += "neighbor        1.0 bin\n"  # 邻居列表截断距离和构建方法
        if neidelay is not None:
            ret += "neigh_modify    delay %d\n" % neidelay  # 邻居列表更新延迟
        ret += "\n"
        # 读取结构文件
        ret += "box          tilt large\n"  # 允许大倾斜角
        ret += "read_data %s\n" % conf_file  # 读取结构数据
        ret += "change_box   all triclinic\n"  # 转换为三斜晶系
        # 设置各元素的质量
        for element, value in element_map.items():
            ret += "mass            %s %f\n" % (
                int(value) + 1,  # LAMMPS中原子类型从1开始
                elements_dict[element.upper()],  # 从元素周期表获取质量
            )
        # 设置势能函数
        if potential_type == "deepmd":
            ret += "pair_style      deepmd %s \n" % atomic_potential  # 深度势能
            ret += "pair_coeff      * *\n"  # DeepMD不需要指定系数
        elif potential_type == "eam":
            ret += "pair_style      eam/alloy\n"  # EAM合金势
            element_string = "".join(str(e) for e in list(element_map.keys()))
            ret += "pair_coeff      * * %s %s\n" % (atomic_potential, element_string)
        ret += "\n"
        # 重置时间步
        ret += "reset_timestep	0\n"
        # 设置热力学输出
        ret += "thermo          ${THERMO_FREQ}\n"  # 输出频率
        ret += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"  # 输出内容
        # 设置轨迹输出
        ret += "dump            1 all custom ${DUMP_FREQ} relax.dump id type xu yu zu\n"  # 输出原子坐标
        # 设置最小化参数
        ret += "min_style cg\n"  # 使用共轭梯度法
        ret += "minimize 1e-15 1e-15 10000 10000\n"  # 能量和力的收敛标准，最大迭代步数
        ret += "undump 1\n"  # 关闭轨迹输出
        return ret


class MDHeating(OP):
    """
    MD加热操作类
    
    功能：将系统从低温（10K）加热到目标温度，使用NPT或NVT系综
    """
    def __init__(self):
        """初始化方法，目前为空"""  
        pass

    @classmethod
    def get_input_sign(cls):
        """
        定义输入参数的类型和结构
        
        返回:
            OPIOSign对象，包含以下输入参数:
            - temp: 目标温度（K）
            - nsteps: 模拟步数
            - ensemble: 系综类型（npt或nvt）
            - pbc: 是否使用周期性边界条件
            - pres: 压力（bar）
            - timestep: 时间步长（ps）
            - neidelay: 邻居列表更新延迟
            - trj_freq: 轨迹输出频率
            - atomic_potential: 势能文件路径
            - structure: 输入结构文件路径
            - element_map: 元素类型映射字典
            - potential_type: 势能类型（deepmd或eam）
            - running_cores: 并行计算使用的核心数
        """
        return OPIOSign(
            {
                "temp": float,  # 目标温度
                "nsteps": int,  # 模拟步数
                "ensemble": str,  # 系综类型
                "pbc": bool,  # 周期性边界条件
                "pres": float,  # 压力
                "timestep": float,  # 时间步长
                "neidelay": int,  # 邻居列表更新延迟
                "trj_freq": int,  # 轨迹输出频率
                "atomic_potential": Artifact(Path),  # 势能文件
                "structure": Artifact(Path),  # 结构文件
                "element_map": dict,  # 元素映射
                "potential_type": str,  # 势能类型
                "running_cores": int,  # 计算核心数
            }
        )

    @classmethod
    def get_output_sign(cls):
        """
        定义输出参数的类型和结构
        
        返回:
            OPIOSign对象，包含以下输出:
            - out_art: 输出文件列表（输入文件、日志文件、轨迹文件）
            - out_structure: 加热后的结构文件
        """
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),  # 输出文件列表
                "out_structure": Artifact(Path),  # 加热后的结构
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        """
        执行加热过程模拟
        
        参数:
            op_in: 输入参数对象，包含温度、系综等信息
        
        返回:
            op_out: 输出参数对象，包含计算结果文件
        
        流程:
            1. 切换到势能文件所在目录
            2. 生成LAMMPS输入文件
            3. 运行LAMMPS进行加热模拟
            4. 读取并转换输出结构
            5. 返回结果文件
        """
        # 保存当前工作目录
        cwd = os.getcwd()
        # 切换到势能文件所在目录
        os.chdir(op_in["atomic_potential"].parent)

        # 生成LAMMPS输入文件内容
        ret = self.make_lmp_input(
            conf_file=op_in["structure"],  # 输入结构文件
            ensemble=op_in["ensemble"],  # 系综类型
            atomic_potential=basename(op_in["atomic_potential"]),  # 势能文件名
            nsteps=op_in["nsteps"],  # 模拟步数
            timestep=op_in["timestep"],  # 时间步长
            neidelay=op_in["neidelay"],  # 邻居列表更新延迟
            trj_freq=op_in["trj_freq"],  # 轨迹输出频率
            temp=op_in["temp"],  # 目标温度
            pres=op_in["pres"],  # 压力
            pbc=op_in["pbc"],  # 周期性边界条件
            element_map=op_in["element_map"],  # 元素映射
            potential_type=op_in["potential_type"],  # 势能类型
        )
        # 写入LAMMPS输入文件
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        # 构建并执行LAMMPS命令
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        # 读取LAMMPS输出文件
        system = dpdata.System(
            "heating.dump",  # LAMMPS轨迹文件
            fmt="lammps/dump",  # 文件格式
            type_map=list(op_in["element_map"].keys()),  # 元素类型映射
            unwrap=True,  # 解包周期性边界
        )
        # 将结构保存为LAMMPS格式，取最后一帧（加热后的结构）
        system.to("lammps/lmp", "output_structure.lammps", frame_idx=-1)
        # 准备输出
        op_out = OPIO(
            {
                "out_art": [Path("input.lammps"), Path("log"), Path("heating.dump")],  # 输出文件列表
                "out_structure": Path("output_structure.lammps"),  # 加热后的结构
            }
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        ensemble: str,
        atomic_potential: str,
        potential_type: str,
        nsteps: int,
        timestep: float,
        neidelay: int,
        trj_freq: int,
        temp: float,
        element_map: dict,
        pres: float = 0.0,
        tau_t: float = 0.5,
        tau_p: float = 0.5,
        pbc: bool = True,
    ):
        """
        生成LAMMPS加热过程的输入文件内容
        
        参数:
            conf_file: 结构文件路径
            ensemble: 系综类型（npt或nvt）
            atomic_potential: 势能文件名
            potential_type: 势能类型（deepmd或eam）
            nsteps: 模拟步数
            timestep: 时间步长（ps）
            neidelay: 邻居列表更新延迟步数
            trj_freq: 轨迹输出频率
            temp: 目标温度（K）
            element_map: 元素类型映射字典
            pres: 压力（bar）
            tau_t: 温度耦合时间常数
            tau_p: 压力耦合时间常数
            pbc: 是否使用周期性边界条件
        
        返回:
            str: LAMMPS输入文件的完整内容
        """
        # 检查NPT系综是否提供了压力参数
        if "npt" in ensemble and pres is None:
            raise RuntimeError("the pressre should be provided for npt ensemble")
        # 设置变量
        ret = "variable        NSTEPS          equal %d\n" % nsteps  # 总步数
        ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq  # 热力学输出频率
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq  # 轨迹输出频率
        ret += "variable        TEMP            equal %f\n" % temp  # 目标温度
        if timestep is not None:
            ret += "variable        TIMESTEP        equal %f\n" % timestep  # 时间步长
        if pres is not None:
            ret += "variable        PRES            equal %f\n" % pres  # 压力
        ret += "variable        TAU_T           equal %f\n" % tau_t  # 温度耦合时间常数
        if pres is not None:
            ret += "variable        TAU_P           equal %f\n" % tau_p  # 压力耦合时间常数
        ret += "\n"
        # 设置单位制为金属单位
        ret += "units           metal\n"
        # 设置边界条件
        if pbc is False:
            ret += "boundary        f f f\n"  # 非周期性边界
        else:
            ret += "boundary        p p p\n"  # 周期性边界
        # 设置原子样式
        ret += "atom_style      atomic\n"
        ret += "\n"
        # 设置邻居列表参数
        ret += "neighbor        1.0 bin\n"  # 邻居列表截断距离和构建方法
        if neidelay is not None:
            ret += "neigh_modify    delay %d\n" % neidelay  # 邻居列表更新延迟
        ret += "\n"
        # 读取结构文件
        ret += "box          tilt large\n"  # 允许大倾斜角
        ret += "read_data %s\n" % conf_file  # 读取结构数据
        ret += "change_box   all triclinic\n"  # 转换为三斜晶系
        # 设置各元素的质量
        for element, value in element_map.items():
            ret += "mass            %s %f\n" % (
                int(value) + 1,  # LAMMPS中原子类型从1开始
                elements_dict[element.upper()],  # 从元素周期表获取质量
            )
        # 设置势能函数
        if potential_type == "deepmd":
            ret += "pair_style      deepmd %s \n" % atomic_potential  # 深度势能
            ret += "pair_coeff      * *\n"  # DeepMD不需要指定系数
        elif potential_type == "eam":
            ret += "pair_style      eam/alloy\n"  # EAM合金势
            element_string = "".join(str(e) for e in list(element_map.keys()))
            ret += "pair_coeff      * * %s %s\n" % (atomic_potential, element_string)
        ret += "\n"
        # 重置时间步
        ret += "reset_timestep	0\n"
        # 设置热力学输出
        ret += "thermo          ${THERMO_FREQ}\n"  # 输出频率
        ret += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"  # 输出内容

        # 设置MD模拟参数
        ret += "timestep ${TIMESTEP}\n"  # 时间步长
        ret += "velocity all create 10 825577 dist gaussian\n"  # 初始化速度（从10K开始）
        # 设置系综
        if pbc is True:
            ret += "fix heating all npt temp 10 ${TEMP} ${TAU_P} iso ${PRES} ${PRES} ${PRES}\n"  # NPT系综
        else:
            ret += "fix heating all nvt temp 10 ${TEMP} ${TAU_T}\n"  # NVT系综
            ret += "velocity all zero angular\n"  # 消除角动量
            ret += "velocity all zero linear\n"  # 消除线动量
        # 设置轨迹输出
        ret += "dump 1 all custom ${DUMP_FREQ} heating.dump id type xu yu zu\n"  # 输出原子坐标
        # 运行模拟
        ret += "run %d\n" % nsteps  # 运行指定步数
        ret += "unfix heating\n"  # 取消系综设置
        ret += "undump 1\n"  # 关闭轨迹输出
        return ret


class MDStabilization(OP):
    """
    MD稳定化操作类
    
    功能：在目标温度下稳定系统，包括结构优化和NVT模拟
    特点：可以选择添加真空层，用于模拟表面体系
    """
    def __init__(self):
        """初始化方法，目前为空"""  
        pass

    @classmethod
    def get_input_sign(cls):
        """
        定义输入参数的类型和结构
        
        返回:
            OPIOSign对象，包含以下输入参数:
            - temp: 目标温度（K）
            - pres: 压力（bar）
            - nsteps: 模拟步数
            - pbc: 是否使用周期性边界条件
            - timestep: 时间步长（ps）
            - neidelay: 邻居列表更新延迟
            - trj_freq: 轨迹输出频率
            - atomic_potential: 势能文件路径
            - structure: 输入结构文件路径
            - element_map: 元素类型映射字典
            - potential_type: 势能类型（deepmd或eam）
            - add_vacuum: 添加真空层的厚度（Å）
            - running_cores: 并行计算使用的核心数
        """
        return OPIOSign(
            {
                "temp": float,  # 目标温度
                "pres": float,  # 压力
                "nsteps": int,  # 模拟步数
                "pbc": bool,  # 周期性边界条件
                "timestep": float,  # 时间步长
                "neidelay": int,  # 邻居列表更新延迟
                "trj_freq": int,  # 轨迹输出频率
                "atomic_potential": Artifact(Path),  # 势能文件
                "structure": Artifact(Path),  # 结构文件
                "element_map": dict,  # 元素映射
                "potential_type": str,  # 势能类型
                "add_vacuum": float,  # 真空层厚度
                "running_cores": int,  # 计算核心数
            }
        )

    @classmethod
    def get_output_sign(cls):
        """
        定义输出参数的类型和结构
        
        返回:
            OPIOSign对象，包含以下输出:
            - out_art: 输出文件列表（输入文件、日志文件、轨迹文件）
            - out_structure: 稳定化后的结构文件
        """
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),  # 输出文件列表
                "out_structure": Artifact(Path),  # 稳定化后的结构
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["atomic_potential"].parent)

        ret = self.make_lmp_input(
            conf_file=op_in["structure"],
            atomic_potential=basename(op_in["atomic_potential"]),
            nsteps=op_in["nsteps"],
            timestep=op_in["timestep"],
            neidelay=op_in["neidelay"],
            trj_freq=op_in["trj_freq"],
            temp=op_in["temp"],
            pbc=op_in["pbc"],
            element_map=op_in["element_map"],
            potential_type=op_in["potential_type"],
            pres=op_in["pres"],
            add_vacuum=op_in["add_vacuum"],
        )
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        system = dpdata.System(
            "stabilization.dump",
            fmt="lammps/dump",
            type_map=list(op_in["element_map"].keys()),
            unwrap=True,
        )
        system.to("lammps/lmp", "output_structure.lammps", frame_idx=-1)
        op_out = OPIO(
            {
                "out_art": [
                    Path("input.lammps"),
                    Path("log"),
                    Path("stabilization.dump"),
                ],
                "out_structure": Path("output_structure.lammps"),
            }
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        atomic_potential: str,
        potential_type: str,
        nsteps: int,
        timestep: float,
        neidelay: int,
        trj_freq: int,
        temp: float,
        element_map: dict,
        tau_t: float = 0.5,
        tau_p: float = 0.5,
        pbc: bool = True,
        pres: float = None,
        add_vacuum: float = 0.0,
    ):
        ret = "variable        NSTEPS          equal %d\n" % nsteps
        ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
        ret += "variable        TEMP            equal %f\n" % temp
        if pres is not None:
            ret += "variable        PRES            equal %f\n" % pres
        ret += "variable        TIMESTEP        equal %f\n" % timestep
        ret += "variable        TAU_T           equal %f\n" % tau_t
        if pres is not None:
            ret += "variable        TAU_P           equal %f\n" % tau_p
        ret += "\n"
        ret += "units           metal\n"
        if pbc is False:
            ret += "boundary        f f f\n"
        else:
            ret += "boundary        p p p\n"
        ret += "atom_style      atomic\n"
        ret += "\n"
        ret += "neighbor        1.0 bin\n"
        if neidelay is not None:
            ret += "neigh_modify    delay %d\n" % neidelay
        ret += "\n"
        ret += "box          tilt large\n"
        ret += "read_data %s\n" % conf_file
        ret += "change_box   all triclinic\n"
        for element, value in element_map.items():
            ret += "mass            %s %f\n" % (
                int(value) + 1,
                elements_dict[element.upper()],
            )
        if potential_type == "deepmd":
            ret += "pair_style      deepmd %s \n" % atomic_potential
            ret += "pair_coeff      * *\n"
        elif potential_type == "eam":
            ret += "pair_style      eam/alloy\n"
            element_string = "".join(str(e) for e in list(element_map.keys()))
            ret += "pair_coeff      * * %s %s\n" % (atomic_potential, element_string)
        ret += "\n"
        ret += "reset_timestep	0\n"
        ret += "thermo          ${THERMO_FREQ}\n"
        ret += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"

        ret += "timestep ${TIMESTEP}\n"

        if pbc:
            ret += "velocity all create ${TEMP} 825577 dist gaussian\n"
            ret += "fix stable_npt all npt temp ${TEMP} ${TEMP} ${TAU_P} iso ${PRES} ${PRES} ${PRES}\n"
            ret += "run %d\n" % nsteps
            ret += "unfix stable_npt\n"
            ret += "min_style cg\n"
            ret += "minimize 1e-15 1e-15 10000 10000\n"
            if add_vacuum > 0.0:
                ret += (
                    "change_box all z delta -{vacuum} {vacuum} boundary p p f\n".format(
                        vacuum=add_vacuum / 2
                    )
                )
            ret += "fix stable_nvt all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
            ret += "velocity all create ${TEMP} 825577 dist gaussian\n"
            ret += "velocity all zero angular\n"
            ret += "velocity all zero linear\n"
            ret += (
                "dump 1 all custom ${DUMP_FREQ} stabilization.dump id type xu yu zu\n"
            )
            ret += "run %d\n" % nsteps
            ret += "unfix stable_nvt\n"
            ret += "undump 1\n"
        else:
            ret += "fix stable_nve all nve\n"
            ret += "fix rescalet all temp/rescale 10 ${TEMP} ${TEMP} 30.0 1.0\n"
            ret += (
                "dump 1 all custom ${DUMP_FREQ} stabilization.dump id type xu yu zu\n"
            )
            ret += "run %d\n" % nsteps
            ret += "unfix stable_nve\n"
            ret += "unfix rescalet\n"
            ret += "undump 1\n"
            ret += "\n"
        return ret


class MDEquilibrium(OP):
    def __init__(self):
        pass

    @classmethod
    def get_input_sign(cls):
        return OPIOSign(
            {
                "temp": float,
                "nsteps": int,
                "pbc": bool,
                "timestep": float,
                "neidelay": int,
                "trj_freq": int,
                "atomic_potential": Artifact(Path),
                "structure": Artifact(Path),
                "element_map": dict,
                "potential_type": str,
                "running_cores": int,
            }
        )

    @classmethod
    def get_output_sign(cls):
        return OPIOSign({"out_art": Artifact(List[Path])})

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        cwd = os.getcwd()
        os.chdir(op_in["atomic_potential"].parent)

        ret = self.make_lmp_input(
            conf_file=op_in["structure"],
            atomic_potential=basename(op_in["atomic_potential"]),
            nsteps=op_in["nsteps"],
            timestep=op_in["timestep"],
            neidelay=op_in["neidelay"],
            trj_freq=op_in["trj_freq"],
            temp=op_in["temp"],
            pbc=op_in["pbc"],
            element_map=op_in["element_map"],
            potential_type=op_in["potential_type"],
        )
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()
        cmd = f"srun -n {op_in['running_cores']} lmp -in input.lammps -l log"
        subprocess.call(cmd, shell=True)
        op_out = OPIO(
            {"out_art": [Path("input.lammps"), Path("log"), Path("equilibrium.dump")]}
        )
        return op_out

    def make_lmp_input(
        self,
        conf_file: str,
        atomic_potential: str,
        potential_type: str,
        nsteps: int,
        timestep: float,
        neidelay: int,
        trj_freq: int,
        temp: float,
        element_map: dict,
        tau_t: float = 0.5,
        pbc: bool = True,
    ):
        ret = "variable        NSTEPS          equal %d\n" % nsteps
        ret += "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
        ret += "variable        TEMP            equal %f\n" % temp
        if timestep is not None:
            ret += "variable        TIMESTEP        equal %f\n" % timestep
        ret += "variable        TAU_T           equal %f\n" % tau_t
        ret += "\n"
        ret += "units           metal\n"
        if pbc is False:
            ret += "boundary        f f f\n"
        else:
            ret += "boundary        p p p\n"
        ret += "atom_style      atomic\n"
        ret += "\n"
        ret += "neighbor        1.0 bin\n"
        if neidelay is not None:
            ret += "neigh_modify    delay %d\n" % neidelay
        ret += "\n"
        ret += "box          tilt large\n"
        ret += "read_data %s\n" % conf_file
        ret += "change_box   all triclinic\n"
        for element, value in element_map.items():
            ret += "mass            %s %f\n" % (
                int(value) + 1,
                elements_dict[element.upper()],
            )
        if potential_type == "deepmd":
            ret += "pair_style      deepmd %s \n" % atomic_potential
            ret += "pair_coeff      * *\n"
        elif potential_type == "eam":
            ret += "pair_style      eam/alloy\n"
            element_string = "".join(str(e) for e in list(element_map.keys()))
            ret += "pair_coeff      * * %s %s\n" % (atomic_potential, element_string)
        ret += "\n"
        ret += "reset_timestep	0\n"
        ret += "thermo          ${THERMO_FREQ}\n"
        ret += "thermo_style    custom step temp pe ke etotal press vol lx ly lz xy xz yz\n"
        ret += "timestep ${TIMESTEP}\n"
        ret += "velocity all create ${TEMP} 825577 dist gaussian\n"
        ret += "fix equilibrium all nvt temp ${TEMP} ${TEMP} ${TAU_T}\n"
        if not pbc:
            ret += "velocity all zero angular\n"
            ret += "velocity all zero linear\n"

        ret += "dump 1 all custom ${DUMP_FREQ} equilibrium.dump id type xu yu zu\n"
        ret += "run %d\n" % nsteps
        ret += "unfix equilibrium\n"
        ret += "undump 1\n"
        return ret
