import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from optCons import findLarger


def pos2idx(x: int, y: int, z: int, nx: int, ny: int, nz: int) -> int:
    return z * (nx * ny) + y * nx + x
    

def idx2pos(idx: int, nx: int, ny: int, nz: int) -> Tuple[int, int, int]:
    z = idx // (nx * ny)
    y = (idx % (nx * ny)) // nx
    x = idx % nx
    return x, y, z

class Model:
    """
    模型类，包含三个方向的尺寸和三维0/1矩阵表示模型状态
    """
    
    def __init__(self, nx: int, ny: int, nz: int):
        """
        初始化模型
        
        Args:
            nx: X方向尺寸
            ny: Y方向尺寸  
            nz: Z方向尺寸
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # 初始化三维0/1矩阵，维度为(nz, ny, nx)，默认全为0
        self.state_matrix = np.zeros((nz, ny, nx), dtype=int)
        self.tempState=-1*np.ones((nz, ny, nx), dtype=int)
    
    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """返回模型的三个方向尺寸 (nx, ny, nz)"""
        return (self.nx, self.ny, self.nz)
    
    def initTempState(self):
        """
        初始化临时状态矩阵
        """
        self.tempState.fill(-1)

    def updateTempState(self, x: int, y: int, z: int, tNow: float, tIdx):
        if self.tempState[z,y,z] != -1:
            return -1
        
        tEndPos, isTNow=findLarger(tIdx, tNow)
        if isTNow:
            return 0
        else:
            if tEndPos==0:
                self.tempState[z,y,x]=0
                return tIdx[0]-tNow
            elif tEndPos==2:
                self.tempState[z,y,x]=0
                return tNow-tIdx[1]
            else:
                self.tempState[z,y,x]=1
                return min(tNow-tIdx[0], tIdx[1]-tNow)



    
    def set_state(self, x: int, y: int, z: int, value: int):
        """
        设置指定位置的状态值
        
        Args:
            x, y, z: 三维坐标
            value: 状态值(0或1)
        """
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            raise IndexError("坐标超出模型范围")
        
        if value not in [0, 1]:
            raise ValueError("状态值必须为0或1")
            
        # 矩阵维度为(nz, ny, nx)，所以访问时为[z, y, x]
        self.state_matrix[z, y, x] = value
    
    def get_state(self, x: int, y: int, z: int) -> int:
        """
        获取指定位置的状态值
        
        Args:
            x, y, z: 三维坐标
            
        Returns:
            状态值(0或1)
        """
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            raise IndexError("坐标超出模型范围")
            
        # 矩阵维度为(nz, ny, nx)，所以访问时为[z, y, x]
        return self.state_matrix[z, y, x]
    
    def fill_region(self, x_start: int, x_end: int, 
                   y_start: int, y_end: int, 
                   z_start: int, z_end: int, value: int):
        """
        填充指定区域
        
        Args:
            x_start, x_end: X方向范围
            y_start, y_end: Y方向范围
            z_start, z_end: Z方向范围
            value: 填充值(0或1)
        """
        if value not in [0, 1]:
            raise ValueError("状态值必须为0或1")
            
        # 矩阵维度为(nz, ny, nx)，所以切片顺序为[z, y, x]
        self.state_matrix[z_start:z_end, y_start:y_end, x_start:x_end] = value
    
    def get_occupied_count(self) -> int:
        """返回状态为1的位置数量"""
        return np.sum(self.state_matrix)
    
    def get_total_count(self) -> int:
        """返回总位置数量"""
        return self.nx * self.ny * self.nz
    
    def get_occupancy_ratio(self) -> float:
        """返回占用率"""
        return self.get_occupied_count() / self.get_total_count()
    
    def clear(self):
        """清空所有状态，将矩阵重置为全0"""
        self.state_matrix.fill(0)
    
    def __str__(self) -> str:
        return f"Model(尺寸: {self.dimensions}, 占用: {self.get_occupied_count()}/{self.get_total_count()})"
    
    def __repr__(self) -> str:
        return self.__str__()

class OperationTime:
    """
    操作时间类，包含三个方向的尺寸和操作时间矩阵
    """
    
    def __init__(self, nx: int, ny: int, nz: int):
        """
        初始化操作时间
        
        Args:
            nx: X方向尺寸
            ny: Y方向尺寸  
            nz: Z方向尺寸
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # 初始化操作时间矩阵，维度为(nx*ny*nz, 2)
        total_elements = nx * ny * nz
        self.time_matrix = np.zeros((total_elements, 2), dtype=float)
    
    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """返回三个方向尺寸 (nx, ny, nz)"""
        return (self.nx, self.ny, self.nz)
    
    @property
    def total_elements(self) -> int:
        """返回总元素数量"""
        return self.nx * self.ny * self.nz
    
    def set_time(self, index: int, start_time: float, end_time: float):
        """
        设置指定索引位置的操作时间
        
        Args:
            index: 元素索引 (0 到 nx*ny*nz-1)
            start_time: 开始时间
            end_time: 结束时间
        """
        if not (0 <= index < self.total_elements):
            raise IndexError(f"索引超出范围，应在0到{self.total_elements-1}之间")
            
        self.time_matrix[index, 0] = start_time
        self.time_matrix[index, 1] = end_time
    
    def set_time_by_coords(self, x: int, y: int, z: int, start_time: float, end_time: float):
        """
        通过三维坐标设置操作时间
        
        Args:
            x, y, z: 三维坐标
            start_time: 开始时间
            end_time: 结束时间
        """
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            raise IndexError("坐标超出范围")
            
        # 将三维坐标转换为一维索引
        index = z * (self.nx * self.ny) + y * self.nx + x
        self.set_time(index, start_time, end_time)
    
    def get_time(self, index: int) -> Tuple[float, float]:
        """
        获取指定索引的操作时间
        
        Args:
            index: 元素索引
            
        Returns:
            (start_time, end_time) 元组
        """
        if not (0 <= index < self.total_elements):
            raise IndexError(f"索引超出范围，应在0到{self.total_elements-1}之间")
            
        return (self.time_matrix[index, 0], self.time_matrix[index, 1])
    
    def to_optimization_variables(self) -> np.ndarray:
        """
        将操作时间转换为优化变量
        
        操作步骤：
        1. 计算第2列减去第1列，形成新的第2列（即计算持续时间）
        2. 将新的矩阵按行展开
        
        Returns:
            一维数组，包含所有优化变量
        """
        # 创建新矩阵的副本
        optimization_matrix = self.time_matrix.copy()
        
        # 第2列减去第1列，形成新的第2列（持续时间）
        optimization_matrix[:, 1] = optimization_matrix[:, 1] - optimization_matrix[:, 0]
        
        # 按行展开矩阵
        flattened = optimization_matrix.flatten()
        
        return flattened
    
    def from_optimization_variables(self, variables: np.ndarray):
        """
        从优化变量恢复操作时间矩阵
        
        Args:
            variables: 一维优化变量数组
        """
        if len(variables) != self.total_elements * 2:
            raise ValueError(f"变量数组长度应为{self.total_elements * 2}，实际为{len(variables)}")
        
        # 将一维数组重塑为矩阵
        reshaped = variables.reshape(self.total_elements, 2)
        
        # 恢复操作时间矩阵：第1列保持不变，第2列为第1列加上持续时间
        self.time_matrix[:, 0] = reshaped[:, 0]  # 开始时间
        self.time_matrix[:, 1] = reshaped[:, 0] + reshaped[:, 1]  # 结束时间 = 开始时间 + 持续时间
    
    def get_duration_matrix(self) -> np.ndarray:
        """
        获取持续时间矩阵（第2列减去第1列）
        
        Returns:
            包含开始时间和持续时间的矩阵
        """
        duration_matrix = self.time_matrix.copy()
        duration_matrix[:, 1] = duration_matrix[:, 1] - duration_matrix[:, 0]
        return duration_matrix
    
    def __str__(self) -> str:
        return f"OperationTime(尺寸: {self.dimensions}, 元素数: {self.total_elements})"
    
    def __repr__(self) -> str:
        return self.__str__()
    


class subModelPara:
    def __init__(self, posStart: Tuple[int, int, int], modelSize: Tuple[int, int, int]):
        self.posStart = posStart
        self.modelSize = modelSize
    
    def getIdxList(self, model: Model) -> List[int]:
        idxList = []
        for z in range(self.posStart[2], self.posStart[2] + self.modelSize[2]):
            for y in range(self.posStart[1], self.posStart[1] + self.modelSize[1]):
                for x in range(self.posStart[0], self.posStart[0] + self.modelSize[0]):
                    idxList.append(pos2idx(x, y, z, model.nx, model.ny, model.nz))
        return idxList