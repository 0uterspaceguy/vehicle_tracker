import numpy as np
from one_euro import OneEuroFilter

from tracker_utils import (convert_bbox_to_z, convert_x_to_bbox, associate_detections_to_trackers)

np.random.seed(0)

class OneEuroBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,
               bbox,
               score,
               class_id,
               min_cutoff: float=1.0,
               beta: float=0.5):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.time = 0 

    self.score = score
    self.class_id = class_id

    self.oef = OneEuroFilter(
      self.time,
      convert_bbox_to_z(bbox),
      min_cutoff=min_cutoff,
      beta=beta
    )

    self.time_since_update = 0
    self.id = OneEuroBoxTracker.count
    OneEuroBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox, score=0, class_id=0):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.time += 1

    self.score = score
    self.class_id = class_id

    self.oef(self.time, convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append((convert_x_to_bbox(self.oef.x_prev)[0], self.score, self.class_id))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.oef.x_prev)[0], self.score, self.class_id
  

class VehicleTracker(object):
  def __init__(self, 
               max_age: int = 1, 
               min_hits: int = 3, 
               iou_weights: float = 0.4,
               distance_weights: float = 0.4,
               area_weights: float = 0.2,
               iou_threshold: float = 0.0,
               distance_threshold: float = 100.0,
               area_threshold: float = 0.5,
               main_threshold: float = 0.5,
               min_cutoff: float = 1.0,
               beta: float = 0.5):

    self.max_age = max_age
    self.min_hits = min_hits

    self.iou_weights = iou_weights
    self.distance_weights = distance_weights
    self.area_weights = area_weights

    self.iou_threshold = iou_threshold
    self.distance_threshold = distance_threshold
    self.area_threshold = area_threshold

    self.main_threshold = main_threshold

    self.min_cutoff = min_cutoff
    self.beta = beta

    self.trackers = []
    self.frame_count = 0

  def update(self, 
             dets=np.empty((0, 4)),
             scores=np.empty((0,)),
             det_class_ids=np.empty((0,))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """

    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5)) 

    to_del = []
    new_boxes = []
    new_scores = []
    new_ids = []
    track_ids = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].class_id]

      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,
                                                                               trks, 
                                                                               det_class_ids, 
                                                                               self.iou_weights,
                                                                               self.distance_weights,
                                                                               self.area_weights,
                                                                               self.iou_threshold,
                                                                               self.distance_threshold,
                                                                               self.area_threshold,
                                                                               self.main_threshold
                                                                               )

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :], scores[m[0]], det_class_ids[m[0]])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = OneEuroBoxTracker(dets[i,:],
                                scores[i],
                                det_class_ids[i],
                                min_cutoff=self.min_cutoff, 
                                beta=self.beta)
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        bbox, score, class_id = trk.get_state()
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          new_boxes.append(bbox)
          new_scores.append(score)
          new_ids.append(class_id)
          track_ids.append(trk.id)

        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)

    if(len(new_boxes)>0):
      return np.array(new_boxes), np.array(new_scores), np.array(new_ids), np.array(track_ids)
    return np.empty((0,4)), np.empty((0,)), np.empty((0,)), np.empty((0,))

