# %%
import json
import math
import torch
import pickle
import numpy as np
import typing as T
from time import time
from einops import rearrange
from rich import print
from rich.progress import track
from pathlib import Path

from pyannote.core import SlidingWindow, SlidingWindowFeature, Annotation, Timeline
from pyannote.pipeline.parameter import ParamDict, Uniform
from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization, batchify
from pyannote.audio.pipelines.speaker_verification import (
    SpeechBrainPretrainedSpeakerEmbedding,
)
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.utils.signal import binarize
from pyannote.audio.pipelines.utils import get_devices

from .base import CADPredictor, Array, FilePath
from config import settings
from utils.annotations import NULL_LABEL, discretize_annotations, get_power_mask


class CADPredictorDiarization(CADPredictor):
    def __init__(self, weights_file: FilePath, params_file: FilePath, use_cache=True):
        super().__init__(weights_file, params_file)
        self._silence_th = settings.AUDIO.SILENCE_TH_DB
        self._fs = settings.AUDIO.SAMPLE_RATE
        self.cache = None
        if use_cache and settings.DIARIZATION.CACHE_ENABLED:
            self.cache = DiarizationCache(
                cache_dir=settings.DIARIZATION.CACHE_DIR,
                max_size=settings.DIARIZATION.CACHE_MAX_SIZE,
                expire_secs=settings.DIARIZATION.CACHE_EXPIRE_HOURS * 3600,
            )

    def load_from_params(self, weights_file: FilePath, params_file: FilePath):
        params = json.load(open(params_file, "r"))

        seg_device, emb_device = get_devices(needs=2)
        print(f"Devices: {seg_device=} | {emb_device=}")

        # Load Segmentation model (steel needed for some introspection params)
        hf_token = str(settings.AUTH.HF_TOKEN)
        model_segmentation = Model.from_pretrained(
            params["seg_model"], use_auth_token=hf_token
        )
        model_segmentation.to(seg_device)
        model_embedding = SpeechBrainPretrainedSpeakerEmbedding(
            params["emb_model"], device=emb_device
        )
        dpipe = CustomSpeakerDiarization(
            model_segmentation,
            model_embedding,
            clustering=params["clustering_algorithm"],
            embedding_batch_size=32,
            embedding_exclude_overlap=params["embedding_exclude_overlap"],
            segmentation_batch_size=32,
        )
        i_params = {
            "clustering": {
                "method": params["clustering_method"],
                "min_cluster_size": params["clustering_min_size"],
                "threshold": params["clustering_threshold"],
            },
            "segmentation": {
                "min_duration_off": params["segmentation_min_duration_off"],
                "threshold": params["segmentation_threshold"],
            },
        }
        dpipe.instantiate(i_params)
        self._dpipe = dpipe

    def predict(self, audio: Array) -> T.Tuple[np.array, np.array, np.array]:
        dpipe = self._dpipe
        audio = torch.tensor(audio)
        audio_f: AudioFile = {"waveform": audio.unsqueeze(0), "sample_rate": self._fs}
        segmentations = None
        embeddings = None
        if self.cache is not None:
            pkl_data = self.cache.find(audio)
            if pkl_data is not None:
                print("[green]Segmentation and embeddings found in CACHE![/green]")
                segmentations, embeddings = pkl_data

        num_speakers, min_speakers, max_speakers = dpipe.set_num_speakers(
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
        )
        if segmentations is None:
            print("[green]Running segmentation...[/green]")
            t0 = time()
            segmentations = dpipe.get_segmentations(audio_f)

            print(f"Segmentation took {time() - t0:.2f} seconds")

        # binarize segmentation
        binarized_segmentations: SlidingWindowFeature = binarize(
            segmentations,
            onset=dpipe.segmentation.threshold,
            initial_state=False,
        )

        if embeddings is None:
            print("[green]Calculating embeddings...[/green]")
            t0 = time()
            embeddings = dpipe.get_embeddings(
                audio_f,
                binarized_segmentations,
                exclude_overlap=dpipe.embedding_exclude_overlap,
            )
            print(f"Embeddings took {time() - t0:.2f} seconds")

        if self.cache is not None:
            print("Saving segmentation and embeddings in cache...")
            self.cache.save(audio, (segmentations, embeddings))

        # Unroll dpipe.speaker_count(segmentations):
        binarized: SlidingWindowFeature = binarize(
            segmentations,
            onset=dpipe.segmentation.threshold,
            offset=None,
            initial_state=False,
        )
        trimmed = Inference.trim(binarized, warm_up=(0.1, 0.1))
        count = Inference.aggregate(
            np.sum(trimmed, axis=-1, keepdims=True),
            frames=dpipe._frames,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)

        print("[green]Running clustering...[/green]")
        t0 = time()
        hard_clusters, _ = dpipe.clustering(
            embeddings=embeddings,
            segmentations=binarized,
            num_clusters=None,
            min_clusters=None,
            max_clusters=None,
        )
        print(f"Clustering took: {time() - t0:.4f} seconds")
        #   hard_clusters: (num_chunks, num_speakers)

        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        print("[green]Running discrete diarization..[/green]")
        t0 = time()
        hard_clusters[inactive_speakers] = -2
        discrete_diarization = dpipe.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        print(f"Discrete diarization took: {time() - t0:.4f} seconds")

        # convert to continuous diarization
        diarization = dpipe.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=dpipe.segmentation.min_duration_off,
        )

        # ----------------------------------------
        # Greater cluster --> Teacher (label "p")
        # Also: overwrite "p" when any other speaker is detected
        # ----------------------------------------
        label_durations = []
        cluster_labels = diarization.labels()
        for cluster_name in cluster_labels:
            label_durations.append(diarization.label_timeline(cluster_name).duration())
        majority_idx = np.argmax(label_durations)

        # assign "p" to greater cluster, "a" to every other cluster
        cluster_rename_map = {
            cluster_name: "p" if idx == majority_idx else "a"
            for idx, cluster_name in enumerate(cluster_labels)
        }
        diarization.rename_labels(cluster_rename_map, copy=False)

        """
        Create "predictions" from diarization, with non-overlapping "p" and "a" clusters
        """
        # Overwrite "p" if there are other speakers detected
        predictions = Annotation()
        segmentation_timeline = Timeline()  # Any non-null class
        for seg in diarization.label_timeline("a").support():
            predictions[seg] = "a"
            segmentation_timeline.add(seg)
        for seg in diarization.label_timeline("p").extrude(
            predictions.label_timeline("a")
        ):
            predictions[seg] = "p"
            segmentation_timeline.add(seg)

        # Null silences & discretize, fill non-silenced gaps with "m"
        hop_s = binarized_segmentations.sliding_window.step  # Could be anything else
        mask_power = get_power_mask(
            audio, self._silence_th, win_len_s=hop_s, hop_len_s=hop_s
        )
        mask_power_ = mask_power.data.squeeze()
        # Discretize using "m" as null label so that all gaps (except silences) are "m"
        y_pred_labels = discretize_annotations(
            predictions, t_step=hop_s, n_samples=mask_power_.shape[0], null_label="m"
        )
        y_pred_labels[~(mask_power_.astype(bool))] = NULL_LABEL

        # Create time axis and return
        t_preds = np.linspace(0, len(audio) / self._fs, len(y_pred_labels))
        y_pred_scores = np.ones_like(y_pred_labels, dtype=np.float16)
        return t_preds, y_pred_labels, y_pred_scores


class DiarizationCache:
    """
    Allows to save segmentation and embeddings for a given audio tensor
    """

    def __init__(self, cache_dir, max_size, expire_secs):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.expire_secs = expire_secs

    def save(self, audio, pickleable_data):
        timestamp_ms = int(time() * 1000)
        # Save audio signature
        np.save(
            self.cache_dir / f"{timestamp_ms}_signature.npy",
            self._audio_signature(audio),
        )
        # Save embeddings
        d_path = self.cache_dir / f"{timestamp_ms}_data.pkl"
        pickle.dump(pickleable_data, open(d_path, "wb"))

        # Clean cache in case it has become too big
        self.clean()

    def clean(self):
        other_suffixes = ("_data.pkl",)
        # Iterate <time>_signature.npy files, and delete also other_suffixes
        for idx, f in enumerate(
            sorted(self.cache_dir.glob("*_signature.npy"))  # sorted: newest to oldest
        ):
            try:
                f_time = f.stem.replace("_signature.npy", "")
                f_time_s = int(f_time) / 1000  # ms -> seconds
                f_life_time_s = time() - f_time_s
                if idx >= self.max_size or not (0 <= f_life_time_s < self.expire_secs):
                    f.unlink()
                    for suffix in other_suffixes:
                        f_other = f.with_name(f"{f_time}{suffix}")
                        f_other.unlink(missing_ok=True)
            except ValueError:
                pass

    def find(self, audio):
        # Compare audio signature to previously saved audios
        # Upon signature match, load precalculated features
        # Assumes previous _clean_cache()
        audio_sig = self._audio_signature(audio)
        for f in self.cache_dir.glob("*_signature.npy"):
            cached_sig = np.load(f)
            if self._check_signatures(audio_sig, cached_sig):

                # Load pickle and return
                fpath = str(f).replace("_signature.npy", "_data.pkl")
                return pickle.load(open(fpath, "rb"))
        return None

    def _audio_signature(self, audio):
        # sum samples each second to create representative but smaller signature
        step = 16000
        audio = np.array(audio)
        pad_len = step - (len(audio) % step)
        reshaped = np.pad(audio, (0, pad_len)).reshape(-1, step)
        return reshaped.sum(axis=1)

    def _check_signatures(self, sig_1, sig_2):
        # Check two signatures created with _cache_signature
        if len(sig_1) != len(sig_2):
            return False
        abs_error = np.sum(np.abs(sig_1 - sig_2)) / len(sig_1)
        return abs_error < 1e-6


class CustomSpeakerDiarization(SpeakerDiarization):
    """
    Customized __init__ to provide instantiated models directly
    """

    def __init__(
        self,
        segmentation_model: Model,
        embedding_model: T.Optional[Model],
        segmentation_duration: float = None,
        segmentation_step: float = 0.1,
        embedding_exclude_overlap: bool = False,
        clustering: str = "HiddenMarkovModelClustering",
        embedding_batch_size: int = 32,
        segmentation_batch_size: int = 32,
        der_variant: dict = None,
        sample_rate: T.Optional[int] = None,
        metric: T.Optional[str] = None,
    ):

        Pipeline.__init__(self)

        self.segmentation_batch_size = segmentation_batch_size
        self.segmentation_duration = (
            segmentation_duration or segmentation_model.specifications.duration
        )
        self.segmentation_step = segmentation_step
        self._segmentation = Inference(
            segmentation_model,
            duration=self.segmentation_duration,
            step=self.segmentation_step * self.segmentation_duration,
            skip_aggregation=True,
            batch_size=self.segmentation_batch_size,
        )
        self._frames: SlidingWindow = self._segmentation.model.introspection.frames

        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.klustering = clustering

        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}

        self.segmentation = ParamDict(
            threshold=Uniform(0.1, 0.9),
            min_duration_off=Uniform(0.0, 1.0),
        )

        self._embedding = embedding_model
        if sample_rate is None or metric is None:
            assert (
                embedding_model is not None
            ), "sample_rate and metric must be provided if embedding_model is not"
            sample_rate = self._embedding.sample_rate
            metric = self._embedding.metric
        self._audio = Audio(sample_rate=sample_rate, mono=True)

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        self.clustering = Klustering.value(metric=metric)

    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
    ):
        """
        Copied from pyannote pipeline, added:
        - track progress bar
        """

        # when optimizing the hyper-parameters of this pipeline with frozen "segmentation_onset",
        # one can reuse the embeddings from the first trial, bringing a massive speed up to
        # the optimization process (and hence allowing to use a larger search space).
        if self.training:

            # we only re-use embeddings if they were extracted based on the same value of the
            # "segmentation_onset" hyperparameter and "embedding_exclude_overlap" parameter.
            cache = file.get("training_cache/embeddings", dict())
            if (
                cache.get("segmentation.threshold", None) == self.segmentation.threshold
                and cache.get("embedding_exclude_overlap", None)
                == self.embedding_exclude_overlap
            ):
                return cache["embeddings"]

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, _ = binary_segmentations.data.shape

        if exclude_overlap:
            # minimum number of samples needed to extract an embedding
            # (a lower number of samples would result in an error)
            min_num_samples = self._embedding.min_num_samples

            # corresponding minimum number of frames
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            # zero-out frames with overlapping speech
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )

        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data, binary_segmentations.sliding_window
            )

        def iter_waveform_and_mask():
            for (chunk, masks), (_, clean_masks) in zip(
                binary_segmentations, clean_segmentations
            ):
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray

                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=duration,
                    mode="pad",
                )
                # waveform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    # mask: (num_frames, ) np.ndarray

                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask

                    yield waveform[None], torch.from_numpy(used_mask)[None]
                    # w: (1, 1, num_samples) torch.Tensor
                    # m: (1, num_frames) torch.Tensor

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )

        embedding_batches = []

        for batch in track(batches):
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

            waveform_batch = torch.vstack(waveforms)
            # (batch_size, 1, num_samples) torch.Tensor

            mask_batch = torch.vstack(masks)
            # (batch_size, num_frames) torch.Tensor

            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )
            # (batch_size, dimension) np.ndarray

            embedding_batches.append(embedding_batch)

        embedding_batches = np.vstack(embedding_batches)

        embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

        # caching embeddings for subsequent trials
        # (see comments at the top of this method for more details)
        if self.training:
            file["training_cache/embeddings"] = {
                "segmentation.threshold": self.segmentation.threshold,
                "embedding_exclude_overlap": self.embedding_exclude_overlap,
                "embeddings": embeddings,
            }

        return embeddings
