from model import OperationTime, Model, subModelPara, idx2pos

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
            continue
        # constraint for supporting

        # constraint for collision-free


        # constraint for SM
        if tIdx[1]>tEnd:
            continue
        # constraint for collision-free

    return satisfied, AElement, ACol, bElement


def consState(initTime, targetModel: Model, tEnd: float, 
              voxelIdx: int):
    # get x,y,z
    x,y,z=idx2pos(voxelIdx, targetModel.nx, targetModel.ny, targetModel.nz)
    # find the first time larger than tEnd
    tEndPos, isTEnd=findLarger(initTime, tEnd)

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
    