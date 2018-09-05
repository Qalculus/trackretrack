import os
import re
from collections import namedtuple
import glob
import argparse

import cv2
import h5py
import numpy as np
import scipy.spatial

from .anms import anms_sdc

try:
    import progressbar

    class Progressbar:
        def __init__(self, _min, _max):
            self._min = _min

            self._indeterminate = _max is None

            widgets = [
                'Frame ', progressbar.Counter(), ' ', progressbar.Percentage(), ' ',
                progressbar.Timer(),
                ' ', progressbar.Bar(initial_value=_min), # Placeholder
                ' ', progressbar.AdaptiveTransferSpeed(unit='frames'), ' ',
                progressbar.AdaptiveETA()
            ]
            self._bar_index = 5

            if self._indeterminate:
                widgets[self._bar_index] = progressbar.AnimatedMarker()
                self.pbar = progressbar.ProgressBar(max_value=progressbar.UnknownLength, widgets=widgets)
            else:
                if _min > 0:
                    # Start indeterminate and switch when we reach min
                    widgets[self._bar_index] = progressbar.AnimatedMarker()
                self.pbar = progressbar.ProgressBar(max_value=_max, widgets=widgets)
            self.pbar.start()

        def update(self, frame_nr):
            if not self._indeterminate and frame_nr > self.pbar.max_value:
                return

            if not self._indeterminate and frame_nr == self._min:
                self.pbar.widgets[self._bar_index] = progressbar.Bar()
            self.pbar.update(frame_nr)

except ImportError:
    class Progressbar:
        def __init__(self, min, max):
            pass

        def update(self, frame_nr):
            pass

Observation = namedtuple("Observation", "frame_nr, pt")

# Assumptions:
# - Frame numbers are increasing, but not necessarily by one
# - Tracks are removed as soon as they fail forward tracking

class Track:
    def __init__(self):
        self.observations = []
        self._frame_obs_map = {}

    def add_observation(self, frame_nr, pt):
        obs = Observation(frame_nr, pt)
        self.observations.append(obs)
        self._frame_obs_map[frame_nr] = obs

    def keep_only_to(self, frame_nr):
        "Keep only observations up to and including frame_nr"
        self.observations = [obs for obs in self.observations if obs.frame_nr <= frame_nr]

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
        return self.observations[0].frame_nr

    @property
    def end(self):
        return self.observations[-1].frame_nr

    @property
    def last(self):
        return self.observations[-1]

    def __getitem__(self, frame_nr):
        return self._frame_obs_map[frame_nr]


NO_ARRAY = np.array([])

debug_list = {}

class TrackRetrack:
    def __init__(self, backtrack_length=3, min_backtrack_distance=0.5, min_track_length=5, min_points=100,
                 min_distance=8, winsize=11):
        if backtrack_length > min_track_length:
            raise ValueError("Minimum track length must be greater than or equal to backtrack length")
        self.backtrack_length = backtrack_length
        self.min_track_length = min_track_length
        self.winsize = (winsize, winsize)
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

        if len(set(self._active_tracks)) < len(self._active_tracks):
            raise ValueError("Active tracks contains duplicates")

        if len(set(self.tracks)) < len(self.tracks):
            raise ValueError("Stored track duplicates!")

    def _find_new_points(self):
        current_num_points = len(self._active_tracks)
        frame_nr, current_frame = self._frame_queue[-1]
        num_to_add = self.min_points - current_num_points
        #detector = cv2.GFTTDetector_create(maxCorners=num_to_add, minDistance=self.min_distance)
        detector = cv2.FastFeatureDetector_create(nonmaxSuppression=True, threshold=15)
        keypoints = detector.detect(current_frame)

        # Non-max suppression
        image_size = self._frame_queue[-1][1].shape[:2]
        keypoints = anms_sdc(keypoints, self.min_points, image_size)

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
        next_pts, status, *_ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_points, NO_ARRAY, winSize=self.winsize)
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
            if len(set(active)) < len(active):
                raise ValueError("Active contains duplicates")

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
                    if track in self.tracks:
                        print([obs.frame_nr for obs in track.observations])
                        print(frame_from_nr, frame_to_nr)
                        print(self._current_frame_nr)
                        print(self.last_retrack(track))
                        print(track in self._backtrack_queue)
                        print(track in already_retracked)
                        print(track in self._active_tracks)
                        raise ValueError("Adding existing track at First! which was added at {}".format(debug_list[track]))
                    self.tracks.append(track)
                    debug_list[track] = ('First', frame_from_nr, self._current_frame_nr)

            if not active:
                continue

            # Track
            prev_points = np.vstack([t[frame_from_nr].pt for t in active])
            prev_points = prev_points.reshape(1, -1, 2)
            next_points, status, *_ = cv2.calcOpticalFlowPyrLK(frame_from, frame_to, prev_points, NO_ARRAY, winSize=self.winsize)

            new_active = []
            for track, new_pt, st in zip(active, next_points[0], status):
                d = np.linalg.norm(new_pt - track[frame_to_nr].pt)
                if st == 1 and d < self.min_backtrack_distance:
                    last_retrack = self._last_retrack.get(track, track.start)
                    if frame_to_nr == last_retrack:
                        if track in self._active_tracks:
                            self._last_retrack[track] = current_frame_nr # Still forward tracking
                        elif track.length >= self.min_track_length:
                            if track in self.tracks:
                                raise ValueError("Adding existing track at Second! which was added at {}".format(debug_list[track]))

                            self.tracks.append(track) # Track has ended but backtracked OK all the way
                            debug_list[track] = ('Second', frame_from_nr, self._current_frame_nr)
                    else:
                        new_active.append(track) # Should be retracked further
                else: # Failed to backtrack
                    try:
                        last_retrack = self._last_retrack[track]
                        track.keep_only_to(last_retrack)
                        if track.length >= self.min_track_length:
                            if track in self.tracks:
                                raise ValueError("Adding existing track at Third! which was added at {}".format(debug_list[track]))
                            self.tracks.append(track)
                            debug_list[track] = ('Third', frame_from_nr, self._current_frame_nr)
                    except KeyError:
                        pass # Retrack failed, and no old valid data -> track dies

                    try:
                        self._backtrack_queue.remove(track)
                    except ValueError:
                        pass # It might be removed already

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

def directory_frames(path, low=0, high=None):
    FILE_REGEXP = re.compile(r'\S*?(\d+)\.\S+')
    for extension in ['.png', '.jpg']:
        files = [f for f in glob.glob(os.path.join(path, '*' + extension)) if FILE_REGEXP.match(f)]
        if files:
            break

    if not files:
        raise ValueError("Directory contains no valid frames")

    def parse_frame_nr(f):
        return int(FILE_REGEXP.match(f).groups()[0])

    files.sort(key=parse_frame_nr)

    max_framenr = parse_frame_nr(files[-1])
    if high is None:
        high = max_framenr
    else:
        high = min(high, max_framenr)

    pbar = Progressbar(low, high)

    for f in files:
        frame_nr = parse_frame_nr(f)
        if low <= frame_nr <= high:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            pbar.update(frame_nr)
            yield frame_nr, img


def video_frames(path, low=0, high=None):
    if not os.path.exists(path):
        raise OSError("Videofile '{}' does not exist".format(path))
    vc = cv2.VideoCapture(path)

    pbar = Progressbar(low, high)

    for frame_nr, _ in enumerate(iter(int, 1)):  # iter(int, 1) is an infinite loop
        res, image = vc.read()

        if not res:
            break

        pbar.update(frame_nr)

        if frame_nr >= low:
            if high is not None and frame_nr > high:
                    break  # Previous frame was last frame
            # Frame was OK and within frame limits, return it
            yield frame_nr, image


def save_tracks_hdf5(tracker, path):
    with h5py.File(path, 'w') as f:
        trackg = f.create_group('tracks')
        for i, t in enumerate(tracker.tracks):
            g = trackg.create_group(str(i))
            frames = [o.frame_nr for o in t.observations]
            points = np.vstack([o.pt for o in t.observations])
            g['frames'] = frames
            g['points'] = points

        f['backtrack_length'] = tracker.backtrack_length
        f['min_distance'] = tracker.min_distance
        f['min_points'] = tracker.min_points
        f['min_length'] = tracker.min_track_length

OUTPUT_HANDLERS = {
    '.h5': save_tracks_hdf5,
    '.hdf5': save_tracks_hdf5
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('videofile')
    parser.add_argument('output')
    parser.add_argument('--min-length', type=int, default=5, help='Minimum track length (frames)')
    parser.add_argument('--backtrack-length', type=int, default=3, help='Number of frames to track back')
    parser.add_argument('--min-points', type=int, default=100, help='Minimum number of points to keep alive per frame')
    parser.add_argument('--min-distance', type=int, default=5, help='Minimum distance (pixels) when creating new points')
    parser.add_argument('--winsize', type=int, default=21, help='LK tracker block size')
    parser.add_argument('--start', type=int, default=0, help='Starting frame (first frame is 0)')
    parser.add_argument('--end', type=int, default=None, help='Ending frame')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output if it exists')
    parser.add_argument('--visualize', action='store_true', help='Plot tracks while tracking')
    args = parser.parse_args()

    if args.visualize:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            parser.error("Visualization requires matplotlib")

    tracker = TrackRetrack(min_track_length=args.min_length,
                           backtrack_length=args.backtrack_length,
                           min_points=args.min_points,
                           min_distance=args.min_distance,
                           winsize=args.winsize)

    in_path = os.path.expanduser(args.videofile)
    out_path = os.path.expanduser(args.output)

    if os.path.exists(out_path) and not args.overwrite:
        parser.error("Output file '{}' already exists".format(out_path))

    _, extension = os.path.splitext(out_path)
    try:

        output_handler = OUTPUT_HANDLERS[extension]
    except KeyError:
        parser.error("No output handler for extension '{}'. Available handlers exist for: {}".format(
            extension, ", ".join(OUTPUT_HANDLERS.keys())
        ))

    if os.path.isdir(in_path):
        print('Loading frames from directory', in_path)
        frame_func = directory_frames
    else:
        print('Loading frames from video file', in_path)
        frame_func = video_frames

    try:
        for frame_nr, im in frame_func(in_path, args.start, args.end):
            tracker.update(frame_nr, im)

            if args.visualize:
                cur_frame_nr, cur_frame = tracker._frame_queue[-1]

                plt.clf()
                plt.imshow(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY), cmap='gray')
                pts = np.vstack([t[cur_frame_nr].pt for t in tracker._active_tracks])
                x, y = pts.T
                plt.scatter(x, y, color='r', marker='x')
                plt.waitforbuttonpress(0.01)

    except (IOError, OSError) as e:
        parser.error(e)

    tracker.finalize()
    output_handler(tracker, out_path)
    print()
    print('Wrote {:d} tracks to {}'.format(len(tracker.tracks), out_path))
