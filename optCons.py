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
        if zPos>0:
            satisfied, AElementTmp, AColTmp, bElementTmp = consAMSupport(initTime, targetModel, tNow, idx)
            if not satisfied:
                return False, AElement, ACol, bElement
            AElement.extend(AElementTmp)
            ACol.extend(AColTmp)
            bElement.extend(bElementTmp)

        # constraint for collision-free
        if zPos<targetModel.nz-1:
            satisfied, AElementTmp, AColTmp, bElementTmp = consAMCollisionFree(initTime, targetModel, tNow, idx)
            if not satisfied:
                return False, AElement, ACol, bElement
            AElement.extend(AElementTmp)
            ACol.extend(AColTmp)
            bElement.extend(bElementTmp)

        # constraint for SM
        if tIdx[1]>tEnd:
            continue
        # constraint for collision-free
        if zPos<targetModel.nz-1 and xPos>0 and xPos<targetModel.nx-1 and yPos>0 and yPos<targetModel.ny-1:
            satisfied, AElementTmp, AColTmp, bElementTmp = consSMCollisionFree(initTime, targetModel, tNow, idx)
            if not satisfied:
                return False, AElement, ACol, bElement
            AElement.extend(AElementTmp)
            ACol.extend(AColTmp)
            bElement.extend(bElementTmp)

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

def consAMSupport(initTime: OperationTime, targetModel: Model, tNow: float, voxelIdx: int):
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
        supportState[0]=1 if targetModel.tempState[z-1,y,x-1]==1 else 0
    if y>0:
        idx=pos2idx(x, y-1, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[1]=targetModel.updateTempState(x, y-1, z-1, tNow, initTime.time_matrix[idx])
        supportState[1]=1 if targetModel.tempState[z-1,y-1,x]==1 else 0
    if x<targetModel.nx-1:
        idx=pos2idx(x+1, y, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[2]=targetModel.updateTempState(x+1, y, z-1, tNow, initTime.time_matrix[idx])
        supportState[2]=1 if targetModel.tempState[z-1,y,x+1]==1 else 0
    if y<targetModel.ny-1:
        idx=pos2idx(x, y+1, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[3]=targetModel.updateTempState(x, y+1, z-1, tNow, initTime.time_matrix[idx])
        supportState[3]=1 if targetModel.tempState[z-1,y+1,x]==1 else 0
    idx=pos2idx(x, y, z-1, targetModel.nx, targetModel.ny, targetModel.nz)
    timeGap[4]=targetModel.updateTempState(x, y, z-1, tNow, initTime.time_matrix[idx])
    supportState[4]=1 if targetModel.tempState[z-1,y,x]==1 else 0
    
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
    
def consAMCollisionFree(initTime: OperationTime, targetModel: Model, tNow: float, voxelIdx: int):
    toolVoxelIdx=targetModel.getAMToolVoxelIdx(voxelIdx)

    AElement=[]
    ACol=[]
    bElement=[]

    satisfied=True

    for idxTool in toolVoxelIdx:
        xTool,yTool,zTool=idx2pos(idxTool, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap=targetModel.updateTempState(xTool,yTool,zTool,tNow,initTime.time_matrix[idxTool])
        
        if targetModel.tempState[zTool,yTool,xTool]<0 or targetModel.tempState[zTool,yTool,xTool]==1:
            return False, [], [], []
        
        if targetModel.tempState[zTool,yTool,xTool]==0:
            AElement.append([-1,1])
            ACol.append([2*idxTool, 2*voxelIdx])
            bElement.append(0)
        else:
            AElement.append([1,1,-1])
            ACol.append([2*idxTool, 2*idxTool+1, 2*voxelIdx])
            bElement.append(0)
    
    return satisfied, AElement, ACol, bElement
    

def consSMCollisionFree(initTime: OperationTime, targetModel: Model, tNow: float, voxelIdx: int):
    x,y,z=idx2pos(voxelIdx, targetModel.nx, targetModel.ny, targetModel.nz)
    
    # 依次检查五个方向是否碰撞
    toolIdxUsed=[]
    toolDirectionUsed=-1
    maxGap=-float('inf')
    for toolDirection in range(5):
        SMToolVoxelIdx, satisfied=targetModel.getSMToolVoxelIdx(voxelIdx, toolDirection)
        if not satisfied:
            # 碰撞
            continue

        miniTimeGap = float('inf')
        for idxTool in SMToolVoxelIdx:
            xTool,yTool,zTool=idx2pos(idxTool, targetModel.nx, targetModel.ny, targetModel.nz)
            timeGap=targetModel.updateTempState(xTool,yTool,zTool,tNow,initTime.time_matrix[idxTool])
            if targetModel.tempState[zTool,yTool,xTool]<0 or targetModel.tempState[zTool,yTool,xTool]==1:
                # 碰撞
                satisfied=False
                break
            if timeGap<miniTimeGap:
                miniTimeGap=timeGap
        
        if not satisfied:
            continue
        if miniTimeGap>maxGap:
            maxGap=miniTimeGap
            toolIdxUsed=SMToolVoxelIdx
            toolDirectionUsed=toolDirection
    
    if len(toolIdxUsed)==0:
        return False, [], [], [], toolDirectionUsed
    
    AElement=[]
    ACol=[]
    bElement=[]
    
    for idxTool in toolIdxUsed:
        if targetModel.tempState[zTool,yTool,xTool]==0:
            AElement.append([-1,1])
            ACol.append([2*idxTool, 2*voxelIdx])
            bElement.append(0)
        else:
            AElement.append([1,1,-1])
            ACol.append([2*idxTool, 2*idxTool+1, 2*voxelIdx])
            bElement.append(0)
    
    return True, AElement, ACol, bElement, toolDirectionUsed
            
    

