from model import OperationTime, Model, subModelPara, idx2pos, pos2idx
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy


def consHMLN(initTime: OperationTime, subModelPara: subModelPara, targetModel: Model, tEnd: float):
    """
    Main constraint function for Hybrid Manufacturing and Logistics Network (HMLN).
    Chooses between parallel and single-thread processing based on problem size.
    """
    x = initTime.to_optimization_variables()
    satisfied = True
    nVar = x.shape[0]

    idxList = subModelPara.getIdxList(targetModel)

    AElement = []
    ACol = []
    bElement = []

    # Use parallel processing for large problems (>1000 indices)
    if len(idxList) > 1000:
        satisfied, AElement, ACol, bElement = process_constraints_parallel_early_exit(
            idxList, targetModel, initTime, tEnd)
    else:
        satisfied, AElement, ACol, bElement = process_constraints_single_thread(
            idxList, targetModel, initTime, tEnd)

    return satisfied, AElement, ACol, bElement


def consState(tIdx, targetModel: Model, tEnd: float, voxelIdx: int):
    """
    Constraint function for voxel state at end time.
    Ensures voxel state matches target state at tEnd.
    """
    # Get x, y, z coordinates
    x, y, z = idx2pos(voxelIdx, targetModel.nx, targetModel.ny, targetModel.nz)
    # Find the first time larger than tEnd
    tEndPos, isTEnd = findLarger(tIdx, tEnd)

    # Construct elements of sparse matrix A
    AElement = []
    ACol = []
    bElement = []

    satisfied = True  # Constraint is satisfied

    if isTEnd:
        satisfied = False  # Some manufacturing happens at tEnd
        return satisfied, AElement, ACol, bElement
    
    if tEndPos == 1:
        # State is solid at tEnd
        if targetModel.state_matrix[z, y, x] == 0:
            satisfied = False  # Target state is empty at tEnd
            return satisfied, AElement, ACol, bElement
        
        AElement.append([1])
        ACol.append([2 * voxelIdx])
        bElement.append(tEnd)

        AElement.append([-1, -1])
        ACol.append([2 * voxelIdx, 2 * voxelIdx + 1])
        bElement.append(-tEnd)
    
    else:
        # State is empty at tEnd
        if targetModel.state_matrix[z, y, x] == 1:
            satisfied = False  # Target state is solid at tEnd
            return satisfied, AElement, ACol, bElement
        
        if tEndPos == 0:
            AElement.append([-1])
            ACol.append([2 * voxelIdx])
            bElement.append(-tEnd)
        else:
            AElement.append([1, 1])
            ACol.append([2 * voxelIdx, 2 * voxelIdx + 1])
            bElement.append(tEnd)
    
    return satisfied, AElement, ACol, bElement


def consAMSupport(initTime: OperationTime, targetModel: Model, tNow: float, voxelIdx: int):
    """
    Constraint function for Additive Manufacturing (AM) support.
    Ensures proper support for AM operations.
    """
    x, y, z = idx2pos(voxelIdx, targetModel.nx, targetModel.ny, targetModel.nz)
    if y == 0:
        return True, [], [], []
    
    supportState, timeGap = checkSupport(x, y, z, tNow, initTime, targetModel)
    if supportState.sum() < 1:
        return False, [], [], []
    
    # Find the index with supportState=1 and maximum timeGap
    maxGap = -float('inf')
    maxIdx = -1
    for i in range(len(supportState)):
        if supportState[i] == 1 and timeGap[i] > maxGap:
            maxGap = timeGap[i]
            maxIdx = i
    
    # Assign support point coordinates based on maxIdx
    if maxIdx == 0:
        # (x-1, y, z-1)
        xSup = x - 1
        ySup = y
        zSup = z - 1
    elif maxIdx == 1:
        # (x, y-1, z-1)
        xSup = x
        ySup = y - 1
        zSup = z - 1
    elif maxIdx == 2:
        # (x+1, y, z-1)
        xSup = x + 1
        ySup = y
        zSup = z - 1
    elif maxIdx == 3:
        # (x, y+1, z-1)
        xSup = x
        ySup = y + 1
        zSup = z - 1
    else:
        # maxIdx == 4, (x, y, z-1)
        xSup = x
        ySup = y
        zSup = z - 1
    
    AElement = []
    ACol = []
    bElement = []
    
    idxSup = pos2idx(xSup, ySup, zSup, targetModel.nx, targetModel.ny, targetModel.nz)
    AElement.append([1, -1])
    ACol.append([2 * idxSup, 2 * voxelIdx])
    bElement.append(0)
    AElement.append([-1, -1, 1])
    ACol.append([2 * idxSup, 2 * idxSup + 1, 2 * voxelIdx])
    bElement.append(0)
    
    return True, AElement, ACol, bElement


def checkSupport(x: int, y: int, z: int, tNow: float, initTime: OperationTime, targetModel: Model):
    """
    Check support points for AM operations.
    
    Args:
        x, y, z: Current voxel coordinates
        tNow: Current time
        initTime: Initial time matrix
        targetModel: Target model
        
    Returns:
        supportState: Support point state (0=no support, 1=support)
        timeGap: Support point time gap
    """
    # Check support points: (x-1,y,z-1), (x,y-1,z-1), (x+1,y,z-1), (x,y+1,z-1), (x,y,z-1)
    supportState = np.zeros(5)
    timeGap = np.zeros(5)
    
    if x > 0:
        idx = pos2idx(x - 1, y, z - 1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[0] = targetModel.updateTempState(x - 1, y, z - 1, tNow, initTime.time_matrix[idx])
        supportState[0] = 1 if targetModel.tempState[z - 1, y, x - 1] == 1 else 0
    if y > 0:
        idx = pos2idx(x, y - 1, z - 1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[1] = targetModel.updateTempState(x, y - 1, z - 1, tNow, initTime.time_matrix[idx])
        supportState[1] = 1 if targetModel.tempState[z - 1, y - 1, x] == 1 else 0
    if x < targetModel.nx - 1:
        idx = pos2idx(x + 1, y, z - 1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[2] = targetModel.updateTempState(x + 1, y, z - 1, tNow, initTime.time_matrix[idx])
        supportState[2] = 1 if targetModel.tempState[z - 1, y, x + 1] == 1 else 0
    if y < targetModel.ny - 1:
        idx = pos2idx(x, y + 1, z - 1, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap[3] = targetModel.updateTempState(x, y + 1, z - 1, tNow, initTime.time_matrix[idx])
        supportState[3] = 1 if targetModel.tempState[z - 1, y + 1, x] == 1 else 0
    idx = pos2idx(x, y, z - 1, targetModel.nx, targetModel.ny, targetModel.nz)
    timeGap[4] = targetModel.updateTempState(x, y, z - 1, tNow, initTime.time_matrix[idx])
    supportState[4] = 1 if targetModel.tempState[z - 1, y, x] == 1 else 0
    
    return supportState, timeGap


def findLarger(manuTime, tCheck: float):
    """
    Find the index of the first element in manuTime that is greater than or equal to tCheck.
    
    Args:
        manuTime: 2D numpy array of manufacturing times
        tCheck: Time threshold to check
        
    Returns:
        tuple: (index, whether it equals tCheck)
    """
    for i, time in enumerate(manuTime):
        if time >= tCheck:
            return i, time == tCheck
    return manuTime.shape[0], False  # If no element >= tCheck is found


def consAMCollisionFree(initTime: OperationTime, targetModel: Model, tNow: float, voxelIdx: int):
    """
    Constraint function for AM collision-free operations.
    Ensures AM tool path is free from collisions.
    """
    toolVoxelIdx = targetModel.getAMToolVoxelIdx(voxelIdx)

    AElement = []
    ACol = []
    bElement = []

    satisfied = True

    for idxTool in toolVoxelIdx:
        xTool, yTool, zTool = idx2pos(idxTool, targetModel.nx, targetModel.ny, targetModel.nz)
        timeGap = targetModel.updateTempState(xTool, yTool, zTool, tNow, initTime.time_matrix[idxTool])
        
        if targetModel.tempState[zTool, yTool, xTool] < 0 or targetModel.tempState[zTool, yTool, xTool] == 1:
            return False, [], [], []
        
        if targetModel.tempState[zTool, yTool, xTool] == 0:
            AElement.append([-1, 1])
            ACol.append([2 * idxTool, 2 * voxelIdx])
            bElement.append(0)
        else:
            AElement.append([1, 1, -1])
            ACol.append([2 * idxTool, 2 * idxTool + 1, 2 * voxelIdx])
            bElement.append(0)
    
    return satisfied, AElement, ACol, bElement


def consSMCollisionFree(initTime: OperationTime, targetModel: Model, tNow: float, voxelIdx: int):
    """
    Constraint function for Subtractive Manufacturing (SM) collision-free operations.
    Checks collision in five directions sequentially and selects the best one.
    """
    x, y, z = idx2pos(voxelIdx, targetModel.nx, targetModel.ny, targetModel.nz)
    
    # Check collision in five directions sequentially
    toolIdxUsed = []
    toolDirectionUsed = -1
    maxGap = -float('inf')
    
    for toolDirection in range(5):
        SMToolVoxelIdx, satisfied = targetModel.getSMToolVoxelIdx(voxelIdx, toolDirection)
        if not satisfied:
            # Collision detected
            continue

        miniTimeGap = float('inf')
        for idxTool in SMToolVoxelIdx:
            xTool, yTool, zTool = idx2pos(idxTool, targetModel.nx, targetModel.ny, targetModel.nz)
            timeGap = targetModel.updateTempState(xTool, yTool, zTool, tNow, initTime.time_matrix[idxTool])
            if targetModel.tempState[zTool, yTool, xTool] < 0 or targetModel.tempState[zTool, yTool, xTool] == 1:
                # Collision detected
                satisfied = False
                break
            if timeGap < miniTimeGap:
                miniTimeGap = timeGap
        
        if not satisfied:
            continue
        if miniTimeGap > maxGap:
            maxGap = miniTimeGap
            toolIdxUsed = SMToolVoxelIdx
            toolDirectionUsed = toolDirection
    
    if len(toolIdxUsed) == 0:
        return False, [], [], [], toolDirectionUsed
    
    AElement = []
    ACol = []
    bElement = []
    
    for idxTool in toolIdxUsed:
        xTool, yTool, zTool = idx2pos(idxTool, targetModel.nx, targetModel.ny, targetModel.nz)
        if targetModel.tempState[zTool, yTool, xTool] == 0:
            AElement.append([-1, 1])
            ACol.append([2 * idxTool, 2 * voxelIdx])
            bElement.append(0)
        else:
            AElement.append([1, 1, -1])
            ACol.append([2 * idxTool, 2 * idxTool + 1, 2 * voxelIdx])
            bElement.append(0)
    
    return True, AElement, ACol, bElement, toolDirectionUsed


def process_constraints_parallel_early_exit(idxList, targetModel, initTime, tEnd, max_workers=None):
    """
    Parallel processing of constraints with true early exit support.
    Uses multiprocessing to handle large constraint sets efficiently.
    """
    AElement = []
    ACol = []
    bElement = []
    
    batch_size = max(1, len(idxList) // (max_workers or 4))
    batches = [idxList[i:i + batch_size] for i in range(0, len(idxList), batch_size)]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch tasks
        futures = [
            executor.submit(process_idx_batch, batch, targetModel, initTime, tEnd)
            for batch in batches
        ]
        
        try:
            # Collect results, cancel other tasks immediately upon error
            for future in as_completed(futures):
                satisfied, AElementTmp, AColTmp, bElementTmp = future.result()
                if not satisfied:
                    # Immediately cancel all unfinished tasks
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    return False, AElement, ACol, bElement
                
                AElement.extend(AElementTmp)
                ACol.extend(AColTmp)
                bElement.extend(bElementTmp)
        
        except Exception as e:
            # Cancel other tasks when exception occurs
            for f in futures:
                if not f.done():
                    f.cancel()
            raise e
    
    return True, AElement, ACol, bElement


def process_idx_batch(idx_batch, targetModel, initTime, tEnd):
    """
    Process a batch of indices for parallel constraint processing.
    """
    AElement_batch = []
    ACol_batch = []
    bElement_batch = []
    
    for idx in idx_batch:
        targetModelNow=copy.deepcopy(targetModel)
        satisfied, AElementTmp, AColTmp, bElementTmp = process_single_idx(idx, targetModelNow, initTime, tEnd)
        if not satisfied:
            return False, None, None, None
        AElement_batch.extend(AElementTmp)
        ACol_batch.extend(AColTmp)
        bElement_batch.extend(bElementTmp)
    
    return True, AElement_batch, ACol_batch, bElement_batch


def process_single_idx(idx, targetModelNow, initTime, tEnd):
    """
    Process constraints for a single index.
    Creates a deep copy of targetModel to avoid interference in parallel processing.
    """
    xPos, yPos, zPos = idx2pos(idx, targetModelNow.nx, targetModelNow.ny, targetModelNow.nz)
    tIdx = initTime.time_matrix[idx]
    
    AElement = []
    ACol = []
    bElement = []
    
    # State constraints
    satisfied, AElementTmp, AColTmp, bElementTmp = consState(tIdx, targetModelNow, tEnd, idx)
    if not satisfied:
        return False, [], [], []
    AElement.extend(AElementTmp)
    ACol.extend(AColTmp)
    bElement.extend(bElementTmp)
    
    # AM constraints
    if tIdx[0] <= tEnd:
        tNow = tIdx[0]
        targetModelNow.initTempState()
        
        # Support constraints
        if zPos > 0:
            satisfied, AElementTmp, AColTmp, bElementTmp = consAMSupport(initTime, targetModelNow, tNow, idx)
            if not satisfied:
                return False, [], [], []
            AElement.extend(AElementTmp)
            ACol.extend(AColTmp)
            bElement.extend(bElementTmp)
        
        # Collision constraints
        if zPos < targetModelNow.nz - 1:
            satisfied, AElementTmp, AColTmp, bElementTmp = consAMCollisionFree(initTime, targetModelNow, tNow, idx)
            if not satisfied:
                return False, [], [], []
            AElement.extend(AElementTmp)
            ACol.extend(AColTmp)
            bElement.extend(bElementTmp)
    
    # SM constraints
    if (tIdx[1] <= tEnd and zPos < targetModelNow.nz - 1 and 
        0 < xPos < targetModelNow.nx - 1 and 0 < yPos < targetModelNow.ny - 1):
        satisfied, AElementTmp, AColTmp, bElementTmp = consSMCollisionFree(initTime, targetModelNow, tNow, idx)
        if not satisfied:
            return False, [], [], []
        AElement.extend(AElementTmp)
        ACol.extend(AColTmp)
        bElement.extend(bElementTmp)
    
    return True, AElement, ACol, bElement


def process_constraints_single_thread(idxList, targetModel, initTime, tEnd):
    """
    Single-threaded processing of constraints for smaller problems.
    More efficient for small constraint sets due to reduced overhead.
    """
    AElement = []
    ACol = []
    bElement = []
    
    for idx in idxList:
        # not need to deepcopy targetModel, because it is single thread
        satisfied, AElementTmp, AColTmp, bElementTmp = process_single_idx(idx, targetModel, initTime, tEnd)
        if not satisfied:
            return False, [], [], []
        AElement.extend(AElementTmp)
        ACol.extend(AColTmp)
        bElement.extend(bElementTmp)

    return True, AElement, ACol, bElement