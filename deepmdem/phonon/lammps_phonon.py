"""
声子计算模块

本模块使用PhonoLAMMPS和Phonopy计算材料的声子性质，包括：
- 声子态密度（DOS）
- 声子能带结构
- 热力学性质

工作流程：
1. 获取原胞结构
2. 使用LAMMPS计算力常数
3. 使用Phonopy计算声子性质
"""
from pathlib import Path  # 文件路径处理
from os.path import basename  # 获取文件名
from dflow.python import OP, OPIO, Artifact, OPIOSign  # dflow工作流相关类
from typing import List  # 类型提示

# import random  # 暂未使用的随机数模块
import os  # 操作系统接口
import dpdata  # 数据处理
from deepmdem.lammps.periodic_table import elements_dict  # 元素周期表
from phonolammps import Phonolammps  # LAMMPS声子计算接口
from phonopy import Phonopy  # 声子分析库
import dpdata  # 数据处理（重复导入）
from pymatgen.core.structure import Structure  # 晶体结构处理


class Phonon(OP):
    """
    声子计算操作类
    
    功能：使用PhonoLAMMPS和Phonopy计算材料的声子性质
    
    主要步骤：
    1. 获取原胞结构
    2. 使用LAMMPS计算力常数
    3. 使用Phonopy计算声子性质（DOS、能带、热力学性质）
    4. 生成可视化图表
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
            - atomic_potential: 势能文件路径
            - structure: 输入结构文件路径
            - element_map: 元素类型映射字典
            - potential_type: 势能类型（deepmd或eam）
            - running_cores: 并行计算使用的核心数
            - supercell: 超胞尺寸，如 [2, 2, 2]
        """
        return OPIOSign(
            {
                "atomic_potential": Artifact(Path),  # 势能文件
                "structure": Artifact(Path),  # 结构文件
                "element_map": dict,  # 元素映射
                "potential_type": str,  # 势能类型
                "running_cores": int,  # 计算核心数
                "supercell": list,  # 超胞尺寸
            }
        )

    @classmethod
    def get_output_sign(cls):
        """
        定义输出参数的类型和结构
        
        返回:
            OPIOSign对象，包含以下输出:
            - out_art: 输出文件列表（输入文件、日志文件、声子参数、
                     能带数据、DOS数据、各种图表等）
        """
        return OPIOSign(
            {
                "out_art": Artifact(List[Path]),  # 输出文件列表
            }
        )

    @OP.exec_sign_check
    def execute(
        self,
        op_in: OPIO,
    ) -> OPIO:
        """
        执行声子计算
        
        参数:
            op_in: 输入参数对象，包含结构、势能等信息
        
        返回:
            op_out: 输出参数对象，包含计算结果文件
        
        流程:
            1. 获取原胞结构
            2. 生成LAMMPS输入文件
            3. 使用PhonoLAMMPS计算力常数
            4. 使用Phonopy计算声子性质
            5. 生成可视化图表
            6. 返回结果文件
        """
        # 保存当前工作目录
        cwd = os.getcwd()
        # 切换到结构文件所在目录
        os.chdir(op_in["structure"].parent)
        
        # 获取输入结构
        input_structure = op_in["structure"]
        # 转换为VASP POSCAR格式
        dpdata.System(input_structure, fmt="auto").to(
            fmt="vasp/poscar", filename="POSCAR"
        )
        # 获取原胞结构
        Structure.from_file("POSCAR").get_primitive_structure().to(
            filename="POSCAR.primitive"
        )
        # 转换为LAMMPS格式
        dpdata.System("POSCAR.primitive", fmt="vasp/poscar").to(
            fmt="lammps/lmp", filename="pcell.lmp"
        )
        pcell_path = Path("pcell.lmp")
        
        # 切换到势能文件所在目录
        os.chdir(op_in["atomic_potential"].parent)
        # 生成LAMMPS输入文件
        ret = self.make_lmp_input(
            conf_file=pcell_path,  # 原胞结构文件
            atomic_potential=basename(op_in["atomic_potential"]),  # 势能文件名
            element_map=op_in["element_map"],  # 元素映射
            potential_type=op_in["potential_type"],  # 势能类型
        )
        # 写入LAMMPS输入文件
        f = open("input.lammps", "w")
        f.write(ret)
        f.close()

        # 创建PhonoLAMMPS对象
        phlammps = Phonolammps(
            "input.lammps",  # LAMMPS输入文件
            supercell_matrix=[  # 超胞变换矩阵
                [op_in["supercell"][0], 0, 0],
                [0, op_in["supercell"][1], 0],
                [0, 0, op_in["supercell"][2]],
            ],
            primitive_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 原胞变换矩阵
        )
        # 获取原胞结构
        unitcell = phlammps.get_unitcell()
        # 计算力常数
        force_constants = phlammps.get_force_constants()
        # 获取超胞矩阵
        supercell_matrix = phlammps.get_supercell_matrix()

        # 创建Phonopy对象
        phonon = Phonopy(unitcell, supercell_matrix)
        # 设置力常数
        phonon.force_constants = force_constants
        # 计算声子网格（用于DOS计算）
        phonon.run_mesh(mesh=[100, 100, 100])  # 100x100x100的q点网格
        # 计算总声子态密度
        phonon.auto_total_dos(filename="total_dos.dat")
        # 计算热力学性质
        phonon.run_thermal_properties()
        # 计算声子能带结构
        phonon.auto_band_structure(write_yaml=True, filename="band.yaml")
        # 保存声子参数
        phonon.save(filename="phonopy_params.yaml")

        dos_plot = phonon.plot_total_DOS()
        dos_plot.savefig("total_dos.png")
        band_plot = phonon.plot_band_structure()
        band_plot.savefig("band.png")
        thermal_plot = phonon.plot_thermal_properties()
        thermal_plot.savefig("thermal.png")
        dos_band_plot = phonon.plot_band_structure_and_dos()
        dos_band_plot.savefig("dos_band.png")

        op_out = OPIO(
            {
                "out_art": [
                    Path("input.lammps"),
                    Path("log"),
                    Path("phonopy_params.yaml"),
                    Path("band.yaml"),
                    Path("total_dos.dat"),
                    Path("total_dos.png"),
                    Path("band.png"),
                    Path("thermal.png"),
                    Path("dos_band.png"),
                ],
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
        ret = "variable        THERMO_FREQ     equal %d\n" % trj_freq
        ret += "variable        DUMP_FREQ       equal %d\n" % trj_freq
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
        ret += "dump            1 all custom ${DUMP_FREQ} relax.dump id type xu yu zu\n"
        ret += "fix 1 all box/relax x 0.0 y 0.0 z 0.0 vmax 0.001"
        ret += "min_style cg\n"
        ret += "minimize 1e-15 1e-15 10000 10000\n"
        ret += "undump 1\n"
        return ret
