"""
MULTEM电子散射模拟模块

本模块使用MULTEM（Multislice Electron Microscopy）库进行电子显微镜成像模拟。
支持STEM（扫描透射电子显微镜）和CBED（会聚束电子衍射）两种模拟类型。
可以使用MD生成的真实声子配置或传统的冻结声子模型。
"""
# 导入晶体结构生成模块（用于测试）
from deepmdem.multem.crystal.Al001_crystal import Al001_crystal
import multem  # MULTEM电子散射模拟库
import dpdata  # 数据处理库
import numpy as np  # 数值计算库
import time  # 时间测量
from deepmdem.multem.dp_to_multem import dp_to_multem  # 数据格式转换
from deepmdem.multem.uilts import (
    potential_pixel,  # 计算势能采样点数
    probe_pixel,  # 计算探针采样点数
    energy2wavelength,  # 能量转波长
    fourier_interpolation,  # 傅里叶插值
    filter_image,  # 图像滤波
)


def run_MULTEM(
    simulation_type: str,
    atoms_conf: str,
    type_map: dict,
    energy: float,
    convergence_angle: float,
    max_scattering_vector: float = None,
    collection_angle: list = None,
    scanning_area: list = None,
    output_size: list = None,
    rmsd: float = 0.0,
    md_phonon: bool = True,
    nphonon: int = 10,
    bwl: bool = True,
):
    """
    执行MULTEM电子散射模拟
    
    参数:
        simulation_type: 模拟类型（"STEM"或"CBED"）
        atoms_conf: 原子配置文件路径（LAMMPS格式）
        type_map: 元素类型映射字典，如 {'Al': 0}
        energy: 电子束能量（keV）
        convergence_angle: 电子束会聚角（mrad）
        max_scattering_vector: 最大散射矢量（1/Å）
        collection_angle: 检测器角度范围列表，如 [[0, 20], [20, 40]]（mrad）
        scanning_area: 扫描区域范围，如 [x0, y0, x1, y1]（相对坐标）
        output_size: 输出图像尺寸，如 [nx, ny]
        rmsd: 原子位移的均方根偏差（Å），用于冻结声子模型
        md_phonon: 是否使用MD生成的声子配置
        nphonon: 冻结声子配置的数量（仅当md_phonon=False时使用）
        bwl: 是否使用带宽限制器
    
    返回:
        tuple: (data, pixel_size)
            data: 模拟结果数据
            pixel_size: 像素尺寸（Å）
    """
    # 记录开始时间
    st = time.time()
    # 检查GPU是否可用
    print("GPU available: %s" % multem.is_gpu_available())

    # 创建MULTEM输入对象和系统配置对象
    input_multislice = multem.Input()
    system_conf = multem.SystemConfiguration()

    # 配置系统参数
    system_conf.precision = "float"  # 使用单精度浮点数
    system_conf.cpu_ncores = 12  # 使用的CPU核心数
    system_conf.cpu_nthread = 1  # 每个核心的线程数
    system_conf.device = "device"  # 计算设备类型
    system_conf.gpu_device = 0  # GPU设备编号

    # 设置模拟实验类型
    input_multislice.simulation_type = simulation_type  # STEM或CBED

    # 设置电子-试样相互作用模型
    input_multislice.interaction_model = "Multislice"  # 使用多层切片算法
    input_multislice.potential_type = "Lobato_0_12"  # 使用Lobato参数化的原子势

    # 设置势能切片参数
    input_multislice.potential_slicing = "dz_Proj"  # 按固定厚度切片
    input_multislice.spec_dz = 2.0  # 切片厚度（Å）
    # input_multislice.potential_slicing = "Planes"  # 按原子平面切片（替代方案）

    # 设置电子-声子相互作用模型
    if md_phonon:
        # 使用MD生成的真实声子配置
        input_multislice.pn_model = "Still_Atom"  # 原子静止，使用MD提供的配置
    else:
        # 使用传统的冻结声子模型
        input_multislice.pn_model = "Frozen_Phonon"  # 冻结声子模型
        input_multislice.pn_coh_contrib = 0  # 相干贡献
        input_multislice.pn_single_conf = False  # 不使用单一配置
        input_multislice.pn_nconf = nphonon  # 声子配置数量
        input_multislice.pn_dim = 110  # 声子维度
        input_multislice.pn_seed = 300183  # 随机数种子

    # 读取MD生成的原子配置
    dp_frame = dpdata.System(atoms_conf, fmt="lammps/lmp")
    # 转换为MULTEM格式
    (
        input_multislice.spec_atoms,  # 原子位置信息
        input_multislice.spec_lx,  # x方向尺寸（Å）
        input_multislice.spec_ly,  # y方向尺寸（Å）
        input_multislice.spec_lz,  # z方向尺寸（Å）
        a, b, c,  # 晶格参数
    ) = dp_to_multem(dp_frame=dp_frame, type_map=type_map, rms3d=rmsd)
    # atoms_1 = np.array(input_multislice.asdict()['spec_atoms'])

    # (
    #     input_multislice.spec_atoms,
    #     input_multislice.spec_lx,
    #     input_multislice.spec_ly,
    #     input_multislice.spec_lz,
    #     a,
    #     b,
    #     c,
    #     input_multislice.spec_dz,
    # ) = Al001_crystal(8, 8, 25, 2, rmsd)
    # atoms_2 = np.array(input_multislice.asdict()['spec_atoms'])

    # import matplotlib.pyplot as plt
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(atoms_1[:, 1], atoms_1[:, 2], atoms_1[:, 3], cmap='Greens');
    # plt.savefig('atoms_1.jpg')

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(atoms_2[:, 1], atoms_2[:, 2], atoms_2[:, 3], cmap='Greens');
    # plt.savefig('atoms_2.jpg')

    # 设置试样厚度
    input_multislice.thick_type = "Through_Thick"  # 遍历厚度
    input_multislice.thick = [x * input_multislice.spec_dz for x in range(0, 100)]  # 厚度列表（0到200Å）

    # 设置显微镜参数
    input_multislice.E_0 = energy  # 电子束能量（keV）
    input_multislice.theta = 0.0  # 倾斜角（度）
    input_multislice.phi = 0.0  # 方位角（度）

    # 设置照明模型
    input_multislice.illumination_model = "Coherent"  # 相干照明
    input_multislice.temporal_spatial_incoh = "Temporal_Spatial"  # 时间和空间非相干性

    # 设置入射波
    input_multislice.iw_type = "Auto"  # 自动设置入射波
    # input_multislice.iw_psi = read_psi_0_multem(input_multislice.nx, input_multislice.ny)  # 自定义入射波（可选）
    input_multislice.iw_x = [input_multislice.spec_lx / 2]  # 电子束x位置（试样中心）
    input_multislice.iw_y = [input_multislice.spec_ly / 2]  # 电子束y位置（试样中心）

    # 设置聚光镜参数
    input_multislice.cond_lens_m = 0  # 聚光镜模式
    input_multislice.cond_lens_c_10 = 0  # 球差系数（mm）
    input_multislice.cond_lens_c_30 = 0  # 六重球差系数（mm）
    input_multislice.cond_lens_c_50 = 0.00  # 十重球差系数（mm）
    input_multislice.cond_lens_c_12 = 0.0  # 二重像散（nm）
    input_multislice.cond_lens_phi_12 = 0.0  # 二重像散角度（度）
    input_multislice.cond_lens_c_23 = 0.0  # 三重像散（nm）
    input_multislice.cond_lens_phi_23 = 0.0  # 三重像散角度（度）
    input_multislice.cond_lens_inner_aper_ang = 0.0  # 内孔径角（mrad）
    input_multislice.cond_lens_outer_aper_ang = convergence_angle  # 外孔径角（mrad）

    # 设置散焦扩散函数（考虑能量色散）
    ti_sigma = multem.iehwgd_to_sigma(32)  # 将半高全宽转换为标准差
    input_multislice.cond_lens_ti_a = 1.0  # 权重因子
    input_multislice.cond_lens_ti_sigma = ti_sigma  # 散焦标准差（Å）
    input_multislice.cond_lens_ti_beta = 0.0  # 偏度参数
    input_multislice.cond_lens_ti_npts = 5  # 采样点数

    # 设置源扩散函数（考虑电子源尺寸）
    si_sigma = multem.hwhm_to_sigma(0.45)  # 将半高全宽转换为标准差
    input_multislice.cond_lens_si_a = 1.0  # 权重因子
    input_multislice.cond_lens_si_sigma = si_sigma  # 源尺寸标准差（mrad）
    input_multislice.cond_lens_si_beta = 0.0  # 偏度参数
    input_multislice.cond_lens_si_rad_npts = 8  # 径向采样点数
    input_multislice.cond_lens_si_azm_npts = 412  # 方位角采样点数

    # 设置零散焦参考
    input_multislice.cond_lens_zero_defocus_type = "First"  # 第一层为零散焦
    input_multislice.cond_lens_zero_defocus_plane = 0  # 零散焦平面

    # 设置探针扫描参数（仅STEM模式）
    if simulation_type == "STEM":
        input_multislice.scanning_type = "Area"  # 区域扫描
        input_multislice.scanning_periodic = True  # 周期性边界条件
        input_multislice.scanning_square_pxs = True  # 使用方形像素
        # 设置扫描区域
        input_multislice.scanning_x0 = scanning_area[0] * input_multislice.spec_lx  # 起始x坐标
        input_multislice.scanning_y0 = scanning_area[1] * input_multislice.spec_ly  # 起始y坐标
        input_multislice.scanning_xe = scanning_area[2] * input_multislice.spec_lx  # 结束x坐标
        input_multislice.scanning_ye = scanning_area[3] * input_multislice.spec_ly  # 结束y坐标
        # 计算最大扫描长度
        max_scanning_length = np.max(
            [
                input_multislice.scanning_xe - input_multislice.scanning_x0,
                input_multislice.scanning_ye - input_multislice.scanning_y0,
            ]
        )
        # 计算扫描点数
        input_multislice.scanning_ns = probe_pixel(
            energy=energy,
            convergence_angle=convergence_angle,
            real_space_length=max_scanning_length,
        )
        # 设置检测器
        input_multislice.detector.type = "Circular"  # 圆形检测器
        input_multislice.detector.cir = collection_angle  # 检测器角度范围

    # 设置x-y方向采样
    if collection_angle is not None:
        # 使用检测器的最大收集角
        max_collection_angle = max(collection_angle, key=lambda item: item[1])[1]
    else:
        # 使用最大散射矢量计算
        max_collection_angle = max_scattering_vector * energy2wavelength(energy) * 1e3
    # 计算x方向采样点数
    input_multislice.nx = potential_pixel(
        energy=energy,
        collection_angle=max_collection_angle,
        real_space_length=input_multislice.spec_lx,
    )
    # 计算y方向采样点数
    input_multislice.ny = potential_pixel(
        energy=energy,
        collection_angle=max_collection_angle,
        real_space_length=input_multislice.spec_ly,
    )
    input_multislice.bwl = bwl  # 带宽限制器

    # 执行模拟
    output_multislice = multem.simulate(system_conf, input_multislice)
    print("Time: %.2f" % (time.time() - st))  # 打印模拟耗时

    # 处理STEM模拟结果
    if simulation_type == "STEM":
        # probe_size = 0.7  # 探针尺寸（Å）
        # pixel_size = (0.1, 0.1)  # 像素尺寸（Å）
        data = []  # 存储所有检测结果
        for i in range(len(output_multislice.data)):  # 遍历所有厚度
            d = []  # 存储当前厚度的所有检测器结果
            for j in range(len(input_multislice.detector.cir)):  # 遍历所有检测器
                image = np.array(output_multislice.data[i].image_tot[j])  # 获取图像
                if output_size is not None:
                    image = fourier_interpolation(image, output_size)  # 傅里叶插值调整尺寸
                # image = filter_image(image, pixel_size, probe_size)  # 图像滤波（可选）
                d.append(image)
            data.append(d)
        pixel_size = max_scanning_length / input_multislice.scanning_ns  # 计算像素尺寸

    # 处理CBED模拟结果
    elif simulation_type == "CBED":
        data = []  # 存储所有衍射图样
        for i in range(len(output_multislice.data)):  # 遍历所有厚度
            m2psi_tot = output_multislice.data[i].m2psi_tot  # 获取衍射强度
            if output_size is not None:
                m2psi_tot = fourier_interpolation(m2psi_tot, output_size)  # 傅里叶插值调整尺寸
            data.append(np.array(m2psi_tot))
        # 计算像素尺寸
        max_collection_angle_in_A = (
            max_collection_angle / energy2wavelength(energy) / 1e3
        )
        pixel_size = max(
            max_collection_angle_in_A * 2 / input_multislice.nx,
            max_collection_angle_in_A * 2 / input_multislice.ny,
        )
        data = np.array(data)
    
    # 保存结果
    fielname = atoms_conf.split(".")[0]  # 获取文件名（不含扩展名）
    np.save(fielname + "multem_out.npy", data)  # 保存为numpy格式
    return data, pixel_size  # 返回数据和像素尺寸
