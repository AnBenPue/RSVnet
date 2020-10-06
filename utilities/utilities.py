import numpy as np
import copy

def getScaledArray(raw_points, high=1.0, low=0.0, mins=None, maxs=None, bycolumn=False):
    """
    Scale the values af an array in order to be between an specific range.

    Parameters
    ----------
    raw_points: N x M array
        Array to be scaled.
    high: float
        Upper bound of the scaled range.
    low: float
        Lower bound of the scaled range.
    bycolumn: bool
        Normalize the array column-wise.
        
    Returns
    ----------
    scaled_input: N x M array
        Array with the scaled values.
    min
        Lower bound of the unscaled range.
    max
        Upper bound of the unscaled range.
    """
    if mins is not None and maxs is not None:
        min = mins
        max = maxs
    else:
        if bycolumn is True:
            min = np.min(raw_points, axis=0)
            max = np.max(raw_points, axis=0)
        else:
            min = np.min(raw_points)
            max = np.max(raw_points)

    rng = max - min
    scaled_input = np.array([high - (((high - low) * (max - raw_point)) / rng) 
                             for raw_point in raw_points]) 
    return scaled_input, min, max

def getUnscaledArray(scaled_points, mins, maxs, high, low):
    """
    Un-scale the values af an array in order to be between the original unscaled
    range.

    Parameters
    ----------
    scaled_points: N x M array
        Array to be unscaled.
    high: float
        Upper bound of the scaled range.
    low: float
        Lower bound of the scaled range.
    min
        Lower bound of the unscaled range.
    max
        Upper bound of the unscaled range.
        
    Returns
    ----------
    unscaled_input: N x M array
        Array with the unscaled values.
    """
    rng = maxs - mins
    unscaled_input = maxs + (scaled_points-high) * rng / ( high - low)
    return  unscaled_input
    
def getPerpendicularVector(v):
    """
    Compute a perpendicular vector to the input one.

    Parameters
    ----------
    v: 1 x 3 array
        Reference vector to which a new perpendicular vector has to be found.
        
    Returns
    ----------
    u: 1 x 3 array
        Perpendicular vector
    """ 
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            u = np.cross(v, [0, 1, 0])
    u = np.cross(v, [1, 0, 0])
    # Scale the vector to have length 1
    u = (1/((np.linalg.norm(u))))*u
    return u

def getMinimalGeodesicDistance(R_hat, R):
    """
    Compute the minimal angle needed to rotate R into R_hat. 

    Parameters
    ----------
    R_hat: 3 x 3 array
        Reference rotation matrix.
    R: 3 x 3 array
        Goal rotation matrix.
        
    Returns
    ----------
    d_R: float
        Rotation angle
    """ 
    R_hat_transpose = np.transpose(R_hat)
    rotation_R_hat_to_R = np.matmul(R_hat_transpose, R)
    trace_value = np.trace(rotation_R_hat_to_R)
    x = (trace_value-1)/2 
    if x > 1:
        x = 1
    elif x < -1:
        x = -1    
    d_R = np.arccos(x)    
    # Convert rad to deg
    d_R = d_R*(180/(2*3.14159265))
    return d_R

def getEuclidianDistance(t_hat, t):
    """
    Compute the euclidian distance between two points. 

    Parameters
    ----------
    t_hat: 1 x 3 array
        Reference point.
    t: 1 x 3 array
        Goal point.
        
    Returns
    ----------
    d_t: float
        Euclidian distance
    """ 
    d_t = np.linalg.norm(t_hat-t)
    return d_t

def getKernelValue(x):
    """
    Apply the un-normalized Gaussian kernel to the value: x. 

    Parameters
    ----------
    x: float
        Input value.
        
    Returns
    ----------
    y: float
        Kernel value.
    """
    y = np.exp((-(x*x)/2))
    return y

def loadT(file_path: str):
    """
    Load a transformation matrix contained in a .txt file 

    Parameters
    ----------
    file_path: str
        Path to the .txt file containing the transformation matrix.
    
    Returns
    ----------
    T: 4 x 4 array
        Transformation matrix
    """ 

    print('INFO: Loading Transformation: ' +  file_path)
    # ToDo: Find out if there is a better way of loading T.
    #--- Fixing ----
    T_stream = open(file_path)
    T_str = T_stream.read(400)
    T_str_clean = T_str.replace("\n"," ")
    T_elements = T_str_clean.split(" ")

    T_list = list()
    
    for i in T_elements:
        if i != '':
            T_list.append(float(i))

    T = np.asarray(T_list)
    T = np.reshape(T_list,[4,4])

    T[0,3] = T[0,3] * 1000
    T[1,3] = T[1,3] * 1000
    T[2,3] = T[2,3] * 1000

    #-------------
    return T

def buildT(translation, rotation):

    T = np.column_stack([rotation, translation])
    T = np.row_stack([T,[0,0,0,1]])
    
    return T

def randomShuffleeList(input_list, num_of_samples=None):
    output_list = copy.deepcopy(input_list)
    np.random.shuffle(output_list)
    return output_list[0:num_of_samples]

def filterListsByIndices(input_lists, indices):
    filtered_lists = list()
    
    lists_length = None

    for l in input_lists:
        current_list = np.asarray(l)
        if lists_length is None:
            lists_length = len(current_list)
        else:
            assert len(current_list) == lists_length
            
        current_list = current_list[indices]
        filtered_lists.append(current_list.tolist())

    return filtered_lists

def getAngleBetweenVectors(u, v):
    c = np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) # -> cosine of the angle
    angle = np.arccos(np.clip(c, -1, 1)) 
    angle = angle*(180/(2*3.14159265))
    return angle