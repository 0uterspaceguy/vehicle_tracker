import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test: np.ndarray, 
              bb_gt: np.ndarray,
              ) -> np.ndarray:
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o) 

def distance_batch(bb_test: np.ndarray, 
                   bb_gt: np.ndarray,
                   ):
  centers_test = np.empty((bb_test.shape[0], 2), dtype=np.float32)
  centers_gt = np.empty((bb_gt.shape[0], 2), dtype=np.float32)

  centers_test[..., 0] = bb_test[..., 0] + (bb_test[..., 2] - bb_test[..., 0]) / 2
  centers_test[..., 1] = bb_test[..., 1] + (bb_test[..., 3] - bb_test[..., 1]) / 2

  centers_gt[..., 0] = bb_gt[..., 0] + (bb_gt[..., 2] - bb_gt[..., 0]) / 2
  centers_gt[..., 1] = bb_gt[..., 1] + (bb_gt[..., 3] - bb_gt[..., 1]) / 2

  distances_matrix = euclidean_distances(centers_test, centers_gt)

  return distances_matrix

def area_sim_batch(bb_test, bb_gt):
  areas_test = np.empty((bb_test.shape[0], 1), dtype=np.float32)
  areas_gt = np.empty((bb_gt.shape[0], 1), dtype=np.float32)

  areas_test[..., 0] = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
  areas_gt[..., 0] = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

  areas_matrix = np.empty((areas_test.shape[0], areas_gt.shape[0]), dtype=np.float32)

  for i, area_test in enumerate(areas_test):
    for j, area_gt in enumerate(areas_gt):
      areas_matrix[i, j] = min(area_test, area_gt) / max(area_test, area_gt)

  return areas_matrix

def prepare_matrix(iou_matrix,
                   distances_matrix,
                   area_matrix,
                   iou_weight = 0.5,
                   distance_weight = 0.3,
                   area_weight = 0.2,
                   iou_thr = 0.0,
                   distance_thr = 100,
                   area_thr = 0.5,
                   ):
  
  distances_matrix = 1 - distances_matrix / np.max(distances_matrix)

  # iou_matrix[iou_matrix < iou_thr] = -1000000
  # distances_matrix[distances_matrix > distance_thr] = 1000000
  # distances_matrix = 1 - distances_matrix / distance_thr
  # area_matrix[area_matrix < area_thr] = -1000000

  # result_matrix =  distances_matrix * distance_weight + area_matrix * area_weight
  result_matrix = iou_matrix * iou_weight  +  distances_matrix * distance_weight + area_matrix * area_weight


  return result_matrix



def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r])



def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
  

def associate_detections_to_trackers(detections,
                                     trackers,
                                     det_class_ids,
                                     iou_weights,
                                     distance_weights,
                                     area_weights,
                                     iou_thr,
                                     distance_thr,
                                     area_thr,
                                     main_thr):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  
  if(len(detections)==0):
    return np.empty((0,2),dtype=int), np.empty((0), dtype=int), np.arange(len(trackers))
  
  
  unmatched_detections = []
  unmatched_trackers = []
  matches = []

  trk_class_ids = np.array([trk[-1] for trk in trackers], dtype=int)
  unique_class_ids = np.unique(np.concatenate((det_class_ids,trk_class_ids)))

  for class_id in unique_class_ids:
    det_class_indices = np.where(det_class_ids == class_id)[0]
    trk_class_indices = np.where(trk_class_ids == class_id)[0]

    class_detections = detections[det_class_indices,:]
    class_trackers = trackers[trk_class_indices]

    if (len(class_detections) == 0) or (len(class_trackers) == 0):
      continue

    iou_weight = iou_weights[class_id]
    distance_weight = distance_weights[class_id]
    area_weight = area_weights[class_id]

    iou_matrix = iou_batch(class_detections, class_trackers)
    distances_matrix = distance_batch(class_detections, class_trackers)
    area_matrix = area_sim_batch(class_detections, class_trackers)

    result_score_matrix = prepare_matrix(iou_matrix=iou_matrix,
                                        distances_matrix=distances_matrix,
                                        area_matrix=area_matrix,
                                        iou_weight=iou_weight,
                                        distance_weight=distance_weight,
                                        area_weight=area_weight,
                                        iou_thr=iou_thr,
                                        distance_thr=distance_thr,
                                        area_thr=area_thr,
                                      )


    if min(result_score_matrix.shape) > 0:
      a = (result_score_matrix > main_thr).astype(np.int32)

      if a.sum(1).max() == 1 and a.sum(0).max() == 1:
          matched_indices = np.stack(np.where(a), axis=1)
      else:
        matched_indices = linear_assignment(-result_score_matrix)

    else:
      matched_indices = np.empty(shape=(0,2))

    for d, det in enumerate(class_detections):
      if(d not in matched_indices[:,0]):
        unmatched_detections.append(d)
    for t, trk in enumerate(class_trackers):
      if(t not in matched_indices[:,1]):
        unmatched_trackers.append(t)

    for m in matched_indices:
      if(result_score_matrix[m[0], m[1]]<main_thr):

        unmatched_detections.append(det_class_indices[m[0]])
        unmatched_trackers.append(trk_class_indices[m[1]])
      else:
        m[0] = det_class_indices[m[0]]
        m[1] = trk_class_indices[m[1]]
        matches.append(m.reshape(1,2))


  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


