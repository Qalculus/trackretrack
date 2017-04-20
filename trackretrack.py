import os
from collections import namedtuple
import argparse

import cv2
import numpy as np
import scipy.spatial

Observation = namedtuple("Observation", "frame_nr, pt")

# Assumptions:
# - Frame numbers are increasing, but not necessarily by one
# - Tracks are removed as soon as they fail forward tracking

class Track:
    def __init__(self):
        self._observations = []
        self._frame_obs_map = {}

    def add_observation(self, frame_nr, pt):
        obs = Observation(frame_nr, pt)
        self._observations.append(obs)
        self._frame_obs_map[frame_nr] = obs

    def keep_only_to(self, frame_nr):
        "Keep only observations up to and including frame_nr"
        self._observations = [obs for obs in self._observations if obs.frame_nr <= frame_nr]

    @property
    def length(self):
        """Number of frames spanned by the track
        
        If the track starts at frame a, and ends (including) frame b its length is L = b - a + 1
        """
        try:
            return self.end - self.start + 1
        except IndexError:
            return 0

    @property
    def start(self):
        return self._observations[0].frame_nr

    @property
    def end(self):
        return self._observations[-1].frame_nr

    @property
    def last(self):
        return self._observations[-1]

    def __getitem__(self, frame_nr):
        return self._frame_obs_map[frame_nr]


NO_ARRAY = np.array([])

class TrackRetrack:
    def __init__(self, backtrack_length=3, min_backtrack_distance=0.5, min_track_length=5, min_points=100, min_distance=8):
        if backtrack_length > min_track_length:
            raise ValueError("Minimum track length must be greater than or equal to backtrack length")
        self.backtrack_length = backtrack_length
        self.min_track_length = min_track_length
        self.min_points = min_points
        self.min_backtrack_distance = min_backtrack_distance
        self.min_distance = min_distance
        self.tracks = []
        self._active_tracks = []
        self._frame_queue = [] #deque(maxlen=self.backtrack_length)
        self._last_retrack = {}
        self._backtrack_queue = []

    def update(self, frame_nr, frame):
        self._frame_queue.append((frame_nr, frame))
        while len(self._frame_queue) > self.backtrack_length + 1: # Need one more frame than backtrack distance
            del self._frame_queue[0]


        if len(self._frame_queue) >= 2:
            self._forward_track()

            assert all(t.last.frame_nr in (frame_nr, frame_nr-1) for t in self._backtrack_queue)

            self._backtrack()

        bad_tracks = [t for t in self._active_tracks if not t.last.frame_nr == frame_nr]
        assert not bad_tracks

        if len(self._active_tracks) < self.min_points:
            self._find_new_points()

    def _find_new_points(self):
        current_num_points = len(self._active_tracks)
        frame_nr, current_frame = self._frame_queue[-1]
        num_to_add = self.min_points - current_num_points
        detector = cv2.GFTTDetector_create(maxCorners=num_to_add, minDistance=self.min_distance)
        keypoints = detector.detect(current_frame)

        if self._active_tracks:
            filtered_keypoints = []
            current_pts = np.vstack([t[frame_nr].pt for t in self._active_tracks])
            kdtree = scipy.spatial.cKDTree(current_pts)

            points = np.array([kp.pt for kp in keypoints])
            distances, indices = kdtree.query(points)

            filtered_keypoints = [kp for kp, d in zip(keypoints, distances) if d >= self.min_distance]
        else:
            filtered_keypoints = keypoints

        for kp in filtered_keypoints:
            t = Track()
            t.add_observation(frame_nr, np.array(kp.pt, dtype='float32'))
            self._active_tracks.append(t)

    def _forward_track(self):
        frame_nr, current_frame = self._frame_queue[-1]
        prev_frame_nr, prev_frame = self._frame_queue[-2]
        assert all(t.last.frame_nr == (frame_nr - 1) for t in self._active_tracks)
        prev_points = np.vstack([t.last.pt for t in self._active_tracks])
        prev_points = prev_points.reshape(1, -1, 2)
        next_pts, status, *_ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, NO_ARRAY)
        next_pts = next_pts[0] # Squeeze first dimension

        new_active = []
        for track, new_pt, st in zip(self._active_tracks, next_pts, status):
            if st == 1:
                track.add_observation(frame_nr, new_pt)
                new_active.append(track)
                if self._needs_backtrack(track):
                    self._backtrack_queue.append(track)
            else:
                if track.length >= self.min_track_length:
                    self._backtrack_queue.append(track)
        self._active_tracks = new_active


    def _backtrack(self):
        if not self._backtrack_queue:
            return

        current_frame_nr, _ = self._frame_queue[-1]
        active = []
        for (frame_from_nr, frame_from), (frame_to_nr, frame_to) in zip(self._frame_queue[::-1], self._frame_queue[-2::-1]):
            # Add tracks going in
            starting = [t for t in self._backtrack_queue
                        if t.last.frame_nr == frame_from_nr and not self.last_retrack(t) == frame_from_nr]
            active.extend(starting)

            # Tracks that ended on a last retrack should not be backtracked
            already_retracked = [t for t in self._backtrack_queue
                                 if t.last.frame_nr == frame_from_nr and self.last_retrack(t) == frame_from_nr]
            for track in already_retracked:
                track.keep_only_to(frame_from_nr)
                if track.length >= self.min_track_length:
                    self.tracks.append(track)

            if not active:
                continue

            # Track
            prev_points = np.vstack([t[frame_from_nr].pt for t in active])
            prev_points = prev_points.reshape(1, -1, 2)
            next_points, status, *_ = cv2.calcOpticalFlowPyrLK(frame_from, frame_to, prev_points, NO_ARRAY)

            new_active = []
            for track, new_pt, st in zip(active, next_points[0], status):
                d = np.linalg.norm(new_pt - track[frame_to_nr].pt)
                if st == 1 and d < self.min_backtrack_distance:
                    last_retrack = self._last_retrack.get(track, track.start)
                    if frame_to_nr == last_retrack:
                        if track in self._active_tracks:
                            self._last_retrack[track] = current_frame_nr # Still forward tracking
                        elif track.length >= self.min_track_length:
                            self.tracks.append(track) # Track has ended but backtracked OK all the way
                    else:
                        new_active.append(track) # Should be retracked further
                else: # Failed to backtrack
                    try:
                        last_retrack = self._last_retrack[track]
                        track.keep_only_to(last_retrack)
                        if track.length >= self.min_track_length:
                            self.tracks.append(track)
                    except KeyError:
                        pass # Retrack failed, and no old valid data -> track dies

                    try:
                        self._active_tracks.remove(track)
                    except ValueError:
                        pass # It might be removed already

            active = new_active
        if active:
            raise RuntimeError("There are still {:d} active backtracks at end of queue".format(len(active)))

        self._backtrack_queue.clear()

    def _needs_backtrack(self, track):
        if track.length < 2:
            return False

        try:
            last_retrack = self._last_retrack[track]
            # Has been retracked before, but was it too long ago?
            elapsed = self._current_frame_nr - last_retrack
            return elapsed >= self.backtrack_length
        except KeyError:
            # Not retracked ever, does it need to now?
            return track.length >= self.backtrack_length + 1

    @property
    def _current_frame_nr(self):
        return self._frame_queue[-1][0]

    def last_retrack(self, track):
        return self._last_retrack.get(track, None)

    def finalize(self):
        if self._active_tracks:
            long_enough = [t for t in self._active_tracks if t.length >= self.backtrack_length]
            self._backtrack_queue.extend(long_enough)
            self._active_tracks.clear()
            self._backtrack()


def video_frames(path, low=0, high=None):
    if not os.path.exists(path):
        raise OSError("Videofile '{}' does not exist".format(path))
    vc = cv2.VideoCapture(path)
    for frame_nr, _ in enumerate(iter(int, 1)):  # iter(int, 1) is an infinite loop
        res, image = vc.read()
        if not res:
            break
        elif frame_nr >= low:
            if high is not None and frame_nr > high:
                    break  # Previous frame was last frame
            # Frame was OK and within frame limits, return it
            yield frame_nr, image



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('videofile')
    parser.add_argument('output')
    parser.add_argument('--min-length', type=int, default=5, help='Minimum track length (frames)')
    parser.add_argument('--backtrack-length', type=int, default=3, help='Number of frames to track back')
    parser.add_argument('--min-points', type=int, default=100, help='Minimum number of points to keep alive per frame')
    parser.add_argument('--min-distance', type=int, default=5, help='Minimum distance (pixels) when creating new points')
    parser.add_argument('--start', type=int, default=0, help='Starting frame (first frame is 0)')
    parser.add_argument('--end', type=int, default=None, help='Ending frame')
    args = parser.parse_args()

    tracker = TrackRetrack(min_track_length=args.min_length,
                           backtrack_length=args.backtrack_length,
                           min_points=args.min_points,
                           min_distance=args.min_distance)

    path = os.path.expanduser(args.videofile)
    try:
        for frame_nr, im in video_frames(path, args.start, args.end):
            print(frame_nr)
            tracker.update(frame_nr, im)

            import matplotlib.pyplot as plt
            if frame_nr >= 2:
                cur_frame_nr, cur_frame = tracker._frame_queue[-1]

                plt.clf()
                plt.imshow(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY), cmap='gray')
                pts = np.vstack([t[cur_frame_nr].pt for t in tracker._active_tracks])
                x, y = pts.T
                plt.scatter(x, y, color='r', marker='x')
                plt.waitforbuttonpress(0.01)

    except (IOError, OSError) as e:
        parser.error(e)

    print('Finalize')
    tracker.finalize()
