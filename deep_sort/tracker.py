# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric #nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        # Tewan
        self.teacher_candidate_ids = []
        self.valid_track = False
        self._features = None
        self._targets = None
        self._active_targets = None
        # Tewan

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def store_features(self):
        self.metric.partial_fit(
            self._features, self._targets, self._active_targets, 
            self.valid_track, store_features=True)
        
    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # detections = Detection(ds_box, cur_score, feature)
        # feature = encoded ROI
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        if not self.valid_track:
            self.tracks = [t for t in self.tracks if not t.is_deleted()]
        else:
            self.tracks = [t for t in self.tracks if t.is_teacher]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        # Tewan: Added all possible teacher ID candidate to active_targets to keep them in nn.samples
        #active_targets.extend(self.teacher_candidate_ids)
        self._features = np.asarray(features)
        self._targets = np.asarray(targets)
        self._active_targets = active_targets

        self.metric.partial_fit(
            self._features, self._targets, self._active_targets, 
            self.valid_track)
        # Tewan
        #self.metric.partial_fit(
        #    np.asarray(features), np.asarray(targets), active_targets)

    def get_fixed_features(self):
        feature_size = {}
        for track, feature in self.metric.samples.items():
            feature_size[track] = len(feature)

        return feature_size
        
    # Tewan
    def update_teacher_candidate_ids(self, teacher_candidate_id):
        if teacher_candidate_id not in self.teacher_candidate_ids:
            self.teacher_candidate_ids.append(teacher_candidate_id)
            track = self._find_track_by_id(track_id=teacher_candidate_id)

            if track:
                track.set_teacher()
                print("Track found.")

    def _find_track_by_id(self, track_id):
        if len(self.tracks) > 0:
            for track in self.tracks:
                if track_id == track.track_id:
                    return track
        return None
    # Tewan

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices]) #confirmed_tracks
            # TODO:Need to add id of teacher ID candidates into targets
            cost_matrix = self.metric.distance(features, targets) #nn_matching cosine distance
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks. Teacher candidaye ids always check for feature
        # Teacher candidates are always confirmed.
        # Check all candidates if not enough teachers had been identified
        if not self.valid_track:
            confirmed_tracks = [
                i for i, t in enumerate(self.tracks) if t.is_confirmed()]
            unconfirmed_tracks = [
                i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        else:
            confirmed_tracks = [
                i for i, t in enumerate(self.tracks) if t.is_teacher]

            #unconfirmed_tracks = [
            #    i for i, t in enumerate(self.tracks) if not t.is_teacher]

        # Associate confirmed tracks using appearance features.
        # matching_cascade(distance_metric, max_distance, 
                           #cascade_depth, tracks, detections, track_indices=None, 
                           #detection_indices=None)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        if not self.valid_track:
            # Associate remaining tracks together with unconfirmed tracks using IOU.
            iou_track_candidates = unconfirmed_tracks + [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update == 1]

            unmatched_tracks_a = [
                k for k in unmatched_tracks_a if
                self.tracks[k].time_since_update != 1]
            # Checks bounding box IOU with existing tracks
            matches_b, unmatched_tracks_b, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, iou_track_candidates, unmatched_detections)

            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
            return matches, unmatched_tracks, unmatched_detections
        return matches_a, unmatched_tracks_a, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        detection_conf = detection.confidence
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection_conf, False))
        self._next_id += 1

    def set_valid_track(self):
        self.valid_track = True