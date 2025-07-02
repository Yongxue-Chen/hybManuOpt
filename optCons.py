from model import OperationTime, Model, subModelPara, idx2pos, pos2idx
import numpy as np

def consHMLN(initTime: OperationTime, subModelPara: subModelPara, targetModel: Model, tEnd: float):
    x = initTime.to_optimization_variables()
    satisfied = True
    nVar = x.shape[0]

    idxList = subModelPara.getIdxList(targetModel)

    AElement=[]
    ACol=[]
    bElement=[]

    for idx in idxList:
        xPos,yPos,zPos=idx2pos(idx, targetModel.nx, targetModel.ny, targetModel.nz)
        tIdx=initTime.time_matrix[idx]
        
        # constraint for states
        satisfied, AElementTmp, AColTmp, bElementTmp = consState(tIdx, targetModel, tEnd, idx)
        if not satisfied:
            return False, AElement, ACol, bElement
        AElement.extend(AElementTmp)
        ACol.extend(AColTmp)
        bElement.extend(bElementTmp)

        # constraint for AM
        if tIdx[0]>tEnd:
            continue # no manu happens before tEnd
        
        tNow=tIdx[0]
        targetModel.initTempState() # initialize temp state matrix, to save state at tNow
        # constraint for supporting
        if yPos>0:
            satisfied, AElementTmp, AColTmp, bElementTmp = consAMSupport(initTime, targetModel, tNow, tEnd, idx)
            if not satisfied:
                return False, AElement, ACol, bElement
            AElement.extend(AElementTmp)
            ACol.extend(AColTmp)
            bElement.extend(bElementTmp)

        # constraint for collision-free


        # constraint for SM
        if tIdx[1]>tEnd:
            continue
        # constraint for collision-free

    return satisfied, AElement, ACol, bElement


def consState(tIdx, targetModel: Model, tEnd: float, voxelIdx: int):
    # get x,y,z
    x,y,z=idx2pos(voxelIdx, targetModel.nx, targetModel.ny, targetModel.nz)
    # find the first time larger than tEnd
    tEndPos, isTEnd=findLarger(tIdx, tEnd)

    # 构造稀疏矩阵A的元素
    AElement=[]
    ACol=[]
    bElement=[]

    satisfied=True # constraint is satisfied

    if isTEnd:
        satisfied=False #some manu happens at tEnd
        return satisfied, AElement, ACol, bElement
    
    if tEndPos == 1:
        # state is solid at tEnd
        if targetModel.state_matrix[z,y,x]==0:
            satisfied=False # target state is empty at tEnd
            return satisfied, AElement, ACol, bElement
        
        AElement.append([1])
        ACol.append([2*voxelIdx])
        bElement.append(tEnd)

        AElement.append([-1,-1])
        ACol.append([2*voxelIdx, 2*voxelIdx+1])
        bElement.append(-tEnd)
    
    else:
        # state is empty at tEnd
        if targetModel.state_matrix[z,y,x]==1:
            satisfied=False # target state is solid at tEnd
            return satisfied, AElement, ACol, bElement
        
        if tEndPos == 0:
            AElement.append([-1])
            ACol.append([2*voxelIdx])
            bElement.append(-tEnd)
        else:
            AElement.append([1,1])
            ACol.append([2*voxelIdx, 2*voxelIdx+1])
            bElement.append(tEnd)
    
    return satisfied, AElement, ACol, bElement

def consAMSupport(initTime: OperationTime, targetModel: Model, tNow: float, tEnd: float, voxelIdx: int):
    x,y,z=idx2pos(voxelIdx, targetModel.nx, targetModel.ny, targetModel.nz)
    if y==0:
        return True, [], [], []
    
    supportState, timeGap=checkSupport(x, y, z, tNow, initTime, targetModel)
    if supportState.sum()<1:
        return False, [], [], []
    
    # 找出supportState中等于1且timeGap最大的序号
    maxGap = -float('inf')
    maxIdx = -1
    for i in range(len(supportState)):
        if supportState[i] == 1 and timeGap[i] > maxGap:
            maxGap = timeGap[i]
            maxIdx = i
    
    # 根据maxIdx的序号给支撑点坐标赋值
    if maxIdx == 0:
        # (x-1, y, z-1)
        xSup = x-1
        ySup = y
        zSup = z-1
    elif maxIdx == 1:
        # (x, y-1, z-1)
        xSup = x
        ySup = y-1
        zSup = z-1
    elif maxIdx == 2:
        # (x+1, y, z-1)
        xSup = x+1
        ySup = y
        zSup = z-1
    elif maxIdx == 3:
        # (x, y+1, z-1)
        xSup = x
        ySup = y+1
        zSup = z-1
    else:
        # maxIdx == 4, (x, y, z-1)
        xSup = x
        ySup = y
        zSup = z-1
    
    AElement=[]
    ACol=[]
    bElement=[]
    
    idxSup=pos2idx(xSup, ySup, zSup, targetModel.nx, targetModel.ny, targetModel.nz)
    AElement.append([1,-1])
    ACol.append([2*idxSup, 2*voxelIdx])
    bElement.append(0)
    AElement.append([-1,-1,1])
    ACol.append([2*idxSup, 2*idxSup+1, 2*voxelIdx])
    bElement.append(0)
    
    return True, AElement, ACol, bElement

def checkSupport(x: int, y: int, z: int, tNow: float, initTime: OperationTime, targetModel: Model):
    """
    检查支撑点
    supportState: 支撑点状态，0表示不支撑，1表示支撑
    timeGap: 支撑点时间差
    """
    # (x-1, y, z-1), (x, y-1, z-1), (x+1, y, z-1), (x, y+1, z-1), (x, y, z-1)
    supportState=np.zeros(5)
    timeGap=np.zeros(5)
    
    if x>0:
        idx=pos2idx(x-1, y, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[0]=targetModel.updateTempState(x-1, y, z-1, tNow, initTime.time_matrix[idx])
        supportState[0]=max(0, targetModel.tempState[z-1,y,x-1])
    if y>0:
        idx=pos2idx(x, y-1, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[1]=targetModel.updateTempState(x, y-1, z-1, tNow, initTime.time_matrix[idx])
        supportState[1]=max(0, targetModel.tempState[z-1,y-1,x])
    if x<targetModel.nx-1:
        idx=pos2idx(x+1, y, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[2]=targetModel.updateTempState(x+1, y, z-1, tNow, initTime.time_matrix[idx])
        supportState[2]=max(0, targetModel.tempState[z-1,y,x+1])
    if y<targetModel.ny-1:
        idx=pos2idx(x, y+1, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[3]=targetModel.updateTempState(x, y+1, z-1, tNow, initTime.time_matrix[idx])
        supportState[3]=max(0, targetModel.tempState[z-1,y+1,x])
    idx=pos2idx(x, y, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
    timeGap[4]=targetModel.updateTempState(x, y, z-1, tNow, initTime.time_matrix[idx])
    supportState[4]=max(0, targetModel.tempState[z-1,y,x])
    
    return supportState, timeGap

def findLarger(manuTime, tCheck: float):
    """
    在manuTime中寻找第一个大于等于tCheck的元素的序号
    
    Args:
        manuTime: 2维numpy数组
        tCheck: 检查的阈值
        
    Returns:
        tuple: (序号, 是否等于tCheck)
    """
    for i, time in enumerate(manuTime):
        if time >= tCheck:
            return i, time == tCheck
    return manuTime.shape[0], False  # 如果没有找到大于等于tCheck的元素
    