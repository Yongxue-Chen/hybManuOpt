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
    
    def __init__(self, nx: int, ny: int, nz: int, tEnd: float):
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
        self.tEnd = tEnd
        
        # 初始化三维0/1矩阵，维度为(nz, ny, nx)，默认全为0
        self.state_matrix = np.zeros((nz, ny, nx), dtype=int)
        self.tempState=-1*np.ones((nz, ny, nx), dtype=int)

        # some tool variables
        self.AMToolParas = {'TipLength': 3, 'TipAngle': 45, 'BodySize': 100}
        self.SMToolParas = {'ToolLength': 10, 'BodyRadius': 100}
    
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
            if self.tempState[z,y,z]==0:
                return tIdx[0]-tNow
            elif self.tempState[z,y,z]==2:
                return tNow-tIdx[1]
            else:
                return min(tNow-tIdx[0], tIdx[1]-tNow)
        
        tNowPos, isTNow=findLarger(tIdx, tNow)
        if isTNow:
            return 0
        else:
            self.tempState[z,y,x]=tNowPos
            if tNowPos==0:
                return tIdx[0]-tNow
            elif tNowPos==2:
                return tNow-tIdx[1]
            else:
                return min(tNow-tIdx[0], tIdx[1]-tNow)

    # def getAMToolVoxelIdx(self, voxelIdx: int):
    #     x,y,z=idx2pos(voxelIdx, self.nx, self.ny, self.nz)
    #     AMToolVoxelIdx=[]
        
    #     idxToolRootStart = pos2idx(0,0,z+self.AMToolLength, self.nx, self.ny, self.nz)

    #     # 从刀尖点到刀根点，刀具直径为1voxel
    #     idxTool=voxelIdx+self.nx*self.ny
    #     while idxTool < min(idxToolRootStart, self.nx*self.ny*self.nz):
    #         AMToolVoxelIdx.append(idxTool)
    #         idxTool+=self.nx*self.ny
        
    #     # 从刀根点高度开始，所有更上层的voxel都被刀具占用
    #     if idxToolRootStart < self.nx*self.ny*self.nz:
    #         for idxTool in range(idxToolRootStart, self.nx*self.ny*self.nz):
    #             AMToolVoxelIdx.append(idxTool)
        
    #     return AMToolVoxelIdx

    def getAMToolVoxelIdx(self, voxelIdx: int):
        """
        获取刀具占用voxel的idx
        """
        x, y, z = idx2pos(voxelIdx, self.nx, self.ny, self.nz)
        AMToolVoxelIdx = []
        
        if self.AMToolParas['TipLength'] > 1:
            for zTool in range(z + 1, min(z + self.AMToolParas['TipLength'], self.nz)):
                rTmp = (zTool - z) * np.tan(self.AMToolParas['TipAngle'] * np.pi / 180)
                xToolStart = int(x - rTmp)  # 转换为整数
                xToolEnd = int(x + rTmp)
                yToolStart = int(y - rTmp)
                yToolEnd = int(y + rTmp)
                
                for xTool in range(max(0, xToolStart), min(self.nx, xToolEnd + 1)):
                    for yTool in range(max(0, yToolStart), min(self.ny, yToolEnd + 1)):
                        if np.linalg.norm([xTool - x, yTool - y]) <= rTmp:
                            AMToolVoxelIdx.append(pos2idx(xTool, yTool, zTool, self.nx, self.ny, self.nz))
        
        if z + self.AMToolParas['TipLength'] < self.nz:
            for zTool in range(z + self.AMToolParas['TipLength'], self.nz):
                # 添加边界检查并转换为整数
                xStart = max(0, int(x - self.AMToolParas['BodySize']/2))
                xEnd = min(self.nx, int(x + self.AMToolParas['BodySize']/2 + 1))
                yStart = max(0, int(y - self.AMToolParas['BodySize']/2))
                yEnd = min(self.ny, int(y + self.AMToolParas['BodySize']/2 + 1))
                
                for xTool in range(xStart, xEnd):
                    for yTool in range(yStart, yEnd):
                        AMToolVoxelIdx.append(pos2idx(xTool, yTool, zTool, self.nx, self.ny, self.nz))
        
        return AMToolVoxelIdx
    
    def getSMToolVoxelIdx(self, voxelIdx: int, toolDirection: int):
        """
        获取减材制造工具占用的voxel索引
        
        Args:
            voxelIdx: 当前体素索引
            toolDirection: 工具方向 (0: 指向z负方向)
        
        Returns:
            SMToolVoxelIdx: 工具占用的体素索引列表
        """
        x, y, z = idx2pos(voxelIdx, self.nx, self.ny, self.nz)
        SMToolVoxelIdx = []

        match toolDirection:
            case 0:
                # 指向z负方向的工具
                
                # 第一部分：切削工具部分(从当前位置向上延伸ToolLength长度)
                idxCutterRoot = voxelIdx + self.SMToolParas['ToolLength'] * self.nx * self.ny
                idxTool = voxelIdx + self.nx * self.ny
                
                while idxTool < min(idxCutterRoot, self.nx * self.ny * self.nz):
                    SMToolVoxelIdx.append(idxTool)
                    idxTool += self.nx * self.ny
                
                # 第二部分：工具主体部分(从切削部分之后到模型顶部的圆柱体)
                if z + self.SMToolParas['ToolLength'] < self.nz:
                    for zTool in range(z + self.SMToolParas['ToolLength'], self.nz):
                        # 添加边界检查和整数转换
                        xStart = max(0, x - self.SMToolParas['BodyRadius'])
                        xEnd = min(self.nx, x + self.SMToolParas['BodyRadius'] + 1)
                        yStart = max(0, y - self.SMToolParas['BodyRadius'])
                        yEnd = min(self.ny, y + self.SMToolParas['BodyRadius'] + 1)
                        
                        for xTool in range(xStart, xEnd):
                            for yTool in range(yStart, yEnd):
                                # 检查是否在圆形范围内
                                if np.linalg.norm([xTool - x, yTool - y]) <= self.SMToolParas['BodyRadius']:
                                    SMToolVoxelIdx.append(pos2idx(xTool, yTool, zTool, self.nx, self.ny, self.nz))
            
            case 1:
                # 指向x正方向的工具
                
                # 第一部分：切削工具部分(从当前位置向x负方向延伸ToolLength长度)
                cutterStart = voxelIdx - min(x, self.SMToolParas['ToolLength'] - 1)
                for idxTool in range(cutterStart, voxelIdx):
                    SMToolVoxelIdx.append(idxTool)
                
                # 第二部分：工具主体部分(可以根据需要添加)
                if x-self.SMToolParas['ToolLength']>=0:
                    if z<=self.SMToolParas['BodyRadius']:
                        # 与底部基础碰撞
                        return [], False
                    
                    yStart=max(0, y-self.SMToolParas['BodyRadius'])
                    yEnd=min(self.ny, y+self.SMToolParas['BodyRadius']+1)
                    zStart=max(0, z-self.SMToolParas['BodyRadius'])
                    zEnd=min(self.nz, z+self.SMToolParas['BodyRadius']+1)

                    for xTool in range(0, x-self.SMToolParas['ToolLength']+1):
                        for yTool in range(yStart, yEnd):
                            for zTool in range(zStart, zEnd):
                                if np.linalg.norm([yTool-y, zTool-z])<=self.SMToolParas['BodyRadius']:
                                    SMToolVoxelIdx.append(pos2idx(xTool, yTool, zTool, self.nx, self.ny, self.nz))

            case 2:
                # 指向x负方向

                # 第一部分：切削工具部分(从当前位置向x正方向延伸ToolLength长度)
                cutterEnd = voxelIdx + min(self.nx-1-x, self.SMToolParas['ToolLength']-1)
                for idxTool in range(voxelIdx, cutterEnd+1):
                    SMToolVoxelIdx.append(idxTool)
                
                # 第二部分：工具主体部分(可以根据需要添加)
                if x+self.SMToolParas['ToolLength']<self.nx:
                    if z<=self.SMToolParas['BodyRadius']:
                        return [], False
                    
                    yStart=max(0, y-self.SMToolParas['BodyRadius'])
                    yEnd=min(self.ny, y+self.SMToolParas['BodyRadius']+1)
                    zStart=max(0, z-self.SMToolParas['BodyRadius'])
                    zEnd=min(self.nz, z+self.SMToolParas['BodyRadius']+1)

                    for xTool in range(x+self.SMToolParas['ToolLength'], self.nx):
                        for yTool in range(yStart, yEnd):
                            for zTool in range(zStart, zEnd):
                                if np.linalg.norm([yTool-y, zTool-z])<=self.SMToolParas['BodyRadius']:
                                    SMToolVoxelIdx.append(pos2idx(xTool, yTool, zTool, self.nx, self.ny, self.nz))

            case 3:
                # 指向y正方向

                # 第一部分：切削工具部分(从当前位置向y负方向延伸ToolLength长度)
                cutterLength=min(y, self.SMToolParas['ToolLength']-1)
                for length in range(1, cutterLength+1):
                    SMToolVoxelIdx.append(voxelIdx-length*self.nx)

                # 第二部分：工具主体部分(可以根据需要添加)
                if y-self.SMToolParas['ToolLength']>=0:
                    if z<=self.SMToolParas['BodyRadius']:
                        return [], False
                    
                    xStart=max(0, x-self.SMToolParas['BodyRadius'])
                    xEnd=min(self.nx, x+self.SMToolParas['BodyRadius']+1)
                    zStart=max(0, z-self.SMToolParas['BodyRadius'])
                    zEnd=min(self.nz, z+self.SMToolParas['BodyRadius']+1)
                    for yTool in range(0, y-self.SMToolParas['ToolLength']+1):
                        for xTool in range(xStart, xEnd):
                            for zTool in range(zStart, zEnd):
                                if np.linalg.norm([xTool-x, zTool-z])<=self.SMToolParas['BodyRadius']:
                                    SMToolVoxelIdx.append(pos2idx(xTool, yTool, zTool, self.nx, self.ny, self.nz))
            
            case 4:
                # 指向y负方向

                # 第一部分：切削工具部分(从当前位置向y正方向延伸ToolLength长度)
                cutterLength=min(self.ny-1-y, self.SMToolParas['ToolLength']-1)
                for length in range(1, cutterLength+1):
                    SMToolVoxelIdx.append(voxelIdx+length*self.nx)

                # 第二部分：工具主体部分(可以根据需要添加)
                if y+self.SMToolParas['ToolLength']<self.ny:
                    if z<=self.SMToolParas['BodyRadius']:
                        return [], False
                    
                    xStart=max(0, x-self.SMToolParas['BodyRadius'])
                    xEnd=min(self.nx, x+self.SMToolParas['BodyRadius']+1)
                    zStart=max(0, z-self.SMToolParas['BodyRadius'])
                    zEnd=min(self.nz, z+self.SMToolParas['BodyRadius']+1)
                    for yTool in range(y+self.SMToolParas['ToolLength'], self.ny):
                        for xTool in range(xStart, xEnd):
                            for zTool in range(zStart, zEnd):
                                if np.linalg.norm([xTool-x, zTool-z])<=self.SMToolParas['BodyRadius']:
                                    SMToolVoxelIdx.append(pos2idx(xTool, yTool, zTool, self.nx, self.ny, self.nz))
            case _:
                # 错误方向
                print("错误方向")
                return [], False
        
        return SMToolVoxelIdx, True
    
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