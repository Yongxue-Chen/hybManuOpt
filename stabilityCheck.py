import numpy as np
from model import Model, OperationTime, idx2pos, pos2idx
from optCons import findLarger

neighborDirection=np.array([
    [-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, 0],
    [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
    [1, -1, 0], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, 0]
])

def getModelAtSM(optTime, smVoxIdx, tEnd, equalCase=0):
    """
    Get model state at a specific spatial-temporal point (SM: Spatial-temporal Marker).
    
    This function generates a binary model representing the state of all voxels 
    at a specific time determined by a reference voxel index.
    
    Args:
        optTime: OperationTime object containing time matrix for all voxels
        smVoxIdx: Index of the reference voxel (spatial-temporal marker)
        tEnd: End time boundary for the operation
        equalCase: Value to assign when times are equal (default: 0)
        
    Returns:
        model: numpy array of shape (nx*ny*nz,) with binary/integer values
               representing the state of each voxel at the reference time
               
    Raises:
        ValueError: If smVoxIdx is out of range or smTime >= tEnd
    """
    # Validate input: check if reference voxel index is within valid range
    if smVoxIdx < 0 or smVoxIdx >= optTime.nx * optTime.ny * optTime.nz:
        raise ValueError("smVoxIdx is out of range")
    
    # Extract the reference time from the specified voxel
    smTime = optTime.time_matrix[smVoxIdx, 1]
    
    # Validate that reference time is within operation boundary
    if smTime >= tEnd:
        raise ValueError("smTime is out of range")
    
    # Initialize model as a n-dimensional array, where n = nx*ny*nz
    # Each element represents the state of corresponding voxel
    model = np.zeros((optTime.nx * optTime.ny * optTime.nz), dtype=int)

    # Iterate through all voxels to determine their state at reference time
    for i in range(optTime.nx * optTime.ny * optTime.nz):
        # Compare current voxel's time with reference time
        largerIdx, equalFlag = findLarger(optTime.time_matrix[i, :], smTime)
        
        if equalFlag:
            # If times are equal, use the specified equal case value
            model[i] = equalCase
        elif largerIdx == 1:
            # If current voxel's time is larger than reference time, set to 1
            model[i] = 1
        # Otherwise, model[i] remains 0 (default initialization)

    return model

def checkStabilityAtSM(model, smVoxIdxList, nx, ny, nz, boxSize=5):
    """
    Checks the stability of the model after an SM operation

    This function determines if the solid structures neighboring a set of voxels
    (acted by SM) are properly supported, i.e., connected to the base of the model (z=0).

    Suppose that the model is stable before the SM operation.
    
    The overall logic is as follows:
    1.  Temporarily remove the voxels acted by SM from the model to simulate the state.
    2.  Identify all separate solid components ("islands") that are neighbors to the SM locations.
    3.  For each component, check if it is connected to the model's base (z=0). This is done by
        treating components as nodes in a graph and exploring their connectivity.
    4.  Components are merged if they are found to be connected to each other.
    5.  The model is considered stable if and only if all components neighboring the SMs are
        grounded (connected to the base).

    Args:
        model (np.ndarray): Binary model array of shape (nx*ny*nz,) where 1 represents a solid voxel.
        smVoxIdxList (list[int]): List of indices for the spatial-temporal marker voxels.
                                  These voxels are assumed to be in the same connected component
                                  and share the same z-coordinate.
        nx (int): Dimension of the model grid in the x-axis.
        ny (int): Dimension of the model grid in the y-axis.
        nz (int): Dimension of the model grid in the z-axis.
        boxSize (int, optional): The padding size to expand the bounding box around the SMs
                                 for the initial component search. Defaults to 5.

    Returns:
        bool: True if the structure is stable at the SMs, False otherwise.
    """

    for smVoxIdx in smVoxIdxList:
        if model[smVoxIdx] !=0:
            raise ValueError("The voxels acted by SM must be 0")

    
    if not smVoxIdxList:
        return True # No voxels to check, considered stable.

    # --- 1. Initialization and Bounding Box Setup ---
    # Validate input: check if all SM voxels are in the same z layer
    smPosList = []
    z_coord=-1
    for smVoxIdx in smVoxIdxList:
        smPosList.append(idx2pos(smVoxIdx, nx, ny, nz))
        if z_coord==-1:
            z_coord=smPosList[-1][2]
        elif smPosList[-1][2]!=z_coord:
            raise ValueError("All SM voxels must be in the same z layer")

    # Get the bounding box of the SM component
    boxRange = np.array([
        [min(p[0] for p in smPosList), max(p[0] for p in smPosList)],
        [min(p[1] for p in smPosList), max(p[1] for p in smPosList)],
        [min(p[2] for p in smPosList), max(p[2] for p in smPosList)]
    ])
    
    # Expand the bounding box to limit the search space for connected components
    boxRange[:, 0] = np.maximum(0, boxRange[:, 0] - boxSize)
    boxRange[0, 1] = min(nx - 1, boxRange[0, 1] + boxSize)
    boxRange[1, 1] = min(ny - 1, boxRange[1, 1] + boxSize)
    boxRange[2, 1] = min(nz - 1, boxRange[2, 1] + boxSize)

    # --- 2. Find Initial Neighboring Solid Components ---
    visited = np.zeros(nx * ny * nz, dtype=int) # 0: unvisited, >0: component index + 1
    numConnectedComponents = 0
    solidIdxList = [] # List to store all solid voxels in the initial components

    # Search for solid neighbors around each SM voxel to find starting points of supporting components
    for x, y, z in smPosList:
        for direction in neighborDirection:
            xNeighbor, yNeighbor, zNeighbor = x + direction[0], y + direction[1], z + direction[2]
            
            # Bounding box check for performance
            if not (boxRange[0,0] <= xNeighbor <= boxRange[0,1] and \
                    boxRange[1,0] <= yNeighbor <= boxRange[1,1] and \
                    boxRange[2,0] <= zNeighbor <= boxRange[2,1]):
                continue

            idxNeighbor = pos2idx(xNeighbor, yNeighbor, zNeighbor, nx, ny, nz)
            if model[idxNeighbor] == 1 and visited[idxNeighbor] == 0:
                numConnectedComponents += 1
                findConnectedComponents(np.array([xNeighbor, yNeighbor, zNeighbor]), numConnectedComponents, model, (nx, ny, nz), visited, solidIdxList, boxRange)

    if numConnectedComponents == 0:
        # No solid neighbors found. Stable only if on the build plate.
        if z_coord == 0:
            return True
        else:
            raise ValueError("The model is not stable before the SM")
    
    if numConnectedComponents==1 and z_coord>0:
        # there is only one component
        # since before SM operation, the model is stable, and the voxel is not at ground
        # the component must be connected to the ground
        # so after SM, the model is stable
        return True

    # --- 3. Analyze and Check Connectivity of Components ---
    isGrounded = np.zeros(numConnectedComponents, dtype=bool)
    componentVoxelQueues = [[] for _ in range(numConnectedComponents)]
    initial_non_empty_indices = set()
    
    # Use a Disjoint Set Union (DSU) structure to track merging of components.
    parent = list(range(numConnectedComponents))
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]

    def unite_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            parent[b] = a
            # If component b was grounded, the new merged component a is now grounded.
            isGrounded[a] = isGrounded[a] or isGrounded[b]
        return a

    # Populate initial component data
    for idx in solidIdxList:
        compIdx = visited[idx] - 1
        componentVoxelQueues[compIdx].append(idx)
        initial_non_empty_indices.add(compIdx)
        if idx // (nx * ny) == 0: # Check z-coordinate
            isGrounded[compIdx] = True
    
    solidIdxList.clear() # No longer needed

    # If all initial components are already connected to the bottom, it's stable.
    if all(isGrounded[find_set(i)] for i in initial_non_empty_indices):
        return True

    # --- 4. Iteratively Explore and Merge Components (BFS-like) ---
    # To keep track of the number of separate, active component sets
    # the "active" here means that the connected components
    numActiveComponents = len(initial_non_empty_indices)
    
    # Continue as long as there are un-grounded components with voxels to check
    # the "active" here means that the components need to be checked
    active_components = list(initial_non_empty_indices)
    while active_components:
        components_to_remove = []
        
        for i in active_components:
            i_root = find_set(i)
            if not componentVoxelQueues[i] or isGrounded[i_root]:
                components_to_remove.append(i)
                continue
            
            idxCheck = componentVoxelQueues[i].pop(0)
            xCheck, yCheck, zCheck = idx2pos(idxCheck, nx, ny, nz)

            # Explore neighbors of the current voxel
            for direction in neighborDirection:
                xNeighbor, yNeighbor, zNeighbor = xCheck + direction[0], yCheck + direction[1], zCheck + direction[2]
                if not (0 <= xNeighbor < nx and 0 <= yNeighbor < ny and 0 <= zNeighbor < nz):
                    continue
                
                idxNeighbor = pos2idx(xNeighbor, yNeighbor, zNeighbor, nx, ny, nz)
                if model[idxNeighbor] == 0:
                    continue
                
                compIdxNeigh = visited[idxNeighbor] - 1

                if compIdxNeigh >= 0: # Neighbor is part of an initial component
                    j_root = find_set(compIdxNeigh)
                    if i_root != j_root:
                        new_root = unite_sets(i_root, j_root)
                        numActiveComponents -= 1
                        
                        # --- Implementation for Comment 1 ---
                        # If all components merge into one and the SM is not on the ground plate,
                        # the structure is guaranteed to be stable.
                        if numActiveComponents == 1 and z_coord > 0:
                            return True
                        
                        # After merging, if the new root component is grounded, we can stop exploring this path.
                        if isGrounded[new_root]:
                            break # Break from neighbor exploration
                else: # Neighbor is a solid voxel not visited before (outside initial bbox)
                    visited[idxNeighbor] = i_root + 1
                    if zNeighbor == 0:
                        isGrounded[i_root] = True
                        break # Found a path to the ground
                    # Add new voxel to its root component's queue for future exploration
                    root_queue_idx = i_root
                    componentVoxelQueues[root_queue_idx].append(idxNeighbor)

            if isGrounded[i_root]:
                # Empty the queue of the now-grounded component to stop its processing
                for member_idx in range(numConnectedComponents):
                    if find_set(member_idx) == i_root:
                       componentVoxelQueues[member_idx].clear()
            
            # --- Implementation for Comment 2 ---
            # If the queue for this sub-component is now empty and its root is not yet grounded,
            # check if the entire root component is fully explored.
            if not isGrounded[i_root] and not componentVoxelQueues[i]:
                is_entire_root_component_exhausted = True
                # Check all other sub-components belonging to the same root
                for j_idx in initial_non_empty_indices:
                    if find_set(j_idx) == i_root and componentVoxelQueues[j_idx]:
                        is_entire_root_component_exhausted = False
                        break
                
                if is_entire_root_component_exhausted:
                    # All queues for this root are empty, but it's not grounded. It's a floating island.
                    return False

        # Safely remove the completed components
        for comp in components_to_remove:
            active_components.remove(comp)

    # Final check: are all components that had voxels initially, now grounded?
    return all(isGrounded[find_set(i)] for i in initial_non_empty_indices)

def findConnectedComponents(startPos, componentIdx, model, modelSize, visited,
                            solidIdxList, boxRange):
    """
    Find all voxels in a connected component of solid material using DFS.
    This search is confined within a specified bounding box.

    Args:
        startPos (np.ndarray): The starting position (x,y,z) for the search.
                               The caller must ensure this is a valid, solid, unvisited voxel.
        componentIdx (int): The identifier for the current component.
        model (np.ndarray): The binary model of the structure.
        modelSize (tuple): The dimensions (nx, ny, nz) of the model.
        visited (np.ndarray): An array to track visited voxels.
        solidIdxList (list): A list to be populated with the indices of voxels in the component.
        boxRange (np.ndarray): The bounding box [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
                               that confines the search.
    """
    nx, ny, nz = modelSize
    stack = [startPos]
    
    # Mark the starting voxel as visited. The caller ensures it's valid.
    startIdx = pos2idx(startPos[0], startPos[1], startPos[2], nx, ny, nz)
    visited[startIdx] = componentIdx
    solidIdxList.append(startIdx)

    while stack:
        currentPos = stack.pop()

        # Explore neighbors
        for direction in neighborDirection:
            neighborPos = currentPos + direction

            # Bounding box check
            if not (boxRange[0, 0] <= neighborPos[0] <= boxRange[0, 1] and
                    boxRange[1, 0] <= neighborPos[1] <= boxRange[1, 1] and
                    boxRange[2, 0] <= neighborPos[2] <= boxRange[2, 1]):
                continue

            idx_neighbor = pos2idx(neighborPos[0], neighborPos[1], neighborPos[2], nx, ny, nz)

            # If the neighbor is a solid and unvisited voxel, add it to the component
            if model[idx_neighbor] == 1 and visited[idx_neighbor] == 0:
                visited[idx_neighbor] = componentIdx
                solidIdxList.append(idx_neighbor)
                stack.append(neighborPos)


    



    