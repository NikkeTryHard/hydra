"""
VocNet GRP pre-training with archive-based data loading.

Trains a Game Result Prediction (GRP) model on .tar.zst archives
containing recording session logs. Uses background prefetching to keep
the GPU fed while the next archive decompresses through the storage layer.
"""

import prelude  # noqa: F401 — must be first (configures logging + warnings)

import os
import random
import torch
import logging
import queue
import threading
from glob import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from model import GRP
from acoustic_core.dataset import Grp
from common import tqdm
from config import config

# Number of archives to decompress concurrently.  Higher = more RAM but
# keeps the GPU fed while archives trickle through the FUSE layer.
PREFETCH_WORKERS = 4
PREFETCH_QUEUE_DEPTH = 6  # how many ready archives to buffer


class GrpArchiveDatasetsIter(IterableDataset):
    """Stream training data from .tar.zst archives with deep prefetch pipeline.

    A pool of background threads continuously decompresses archives into a
    queue.  The main thread pulls fully-processed archive buffers from the
    queue and yields individual samples to the DataLoader.  This keeps GPU
    utilisation high even when each archive takes minutes to read through
    an encrypted FUSE mount.
    """

    def __init__(self, archive_list, log_batch_size=500, cycle=False):
        super().__init__()
        self.archive_list = list(archive_list)
        self.log_batch_size = log_batch_size
        self.cycle = cycle
        self.buffer = []
        self.iterator = None

    def _load_archive(self, archive_path):
        """Load and process a single .tar.zst archive (runs in bg thread)."""
        raw_logs = Grp.extract_tar_zst(archive_path)
        results = []

        # Process in batches to avoid holding the entire raw log set in memory
        for start in range(0, len(raw_logs), self.log_batch_size):
            batch_logs = raw_logs[start : start + self.log_batch_size]
            games = Grp.load_raw_logs(batch_logs)
            for game in games:
                feature = game.take_feature()
                rank_by_player = game.take_rank_by_player()
                for i in range(feature.shape[0]):
                    inputs_seq = torch.as_tensor(feature[: i + 1], dtype=torch.float64)
                    results.append((inputs_seq, rank_by_player))
        return results

    def _producer(self, archives, ready_queue, stop_event):
        """Background: submit archives to thread pool, put results in queue."""
        with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
            futures = []
            for archive_path in archives:
                if stop_event.is_set():
                    break
                fut = pool.submit(self._load_archive, archive_path)
                futures.append(fut)

            for fut in futures:
                if stop_event.is_set():
                    break
                try:
                    data = fut.result()
                    ready_queue.put(data)
                except Exception as e:
                    logging.error(f"archive load failed: {e}")
                    ready_queue.put([])  # empty sentinel to keep count

            ready_queue.put(None)  # end-of-epoch sentinel

    def build_iter(self):
        stop_event = threading.Event()
        try:
            while True:
                # Sort archives smallest-first so we get the first training
                # batches to the GPU quickly while larger archives prefetch.
                archives = sorted(self.archive_list, key=lambda p: os.path.getsize(p))
                random.shuffle(archives[3:])  # randomise after the 3 smallest

                ready_queue = queue.Queue(maxsize=PREFETCH_QUEUE_DEPTH)

                producer_thread = threading.Thread(
                    target=self._producer,
                    args=(archives, ready_queue, stop_event),
                    daemon=True,
                )
                producer_thread.start()

                while True:
                    self.buffer = ready_queue.get()
                    if self.buffer is None:
                        break  # end of epoch

                    buf_size = len(self.buffer)
                    if buf_size > 0:
                        for i in random.sample(range(buf_size), buf_size):
                            yield self.buffer[i]
                    self.buffer = []

                producer_thread.join(timeout=5)

                if not self.cycle:
                    break
        finally:
            stop_event.set()

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def collate(batch):
    inputs = []
    lengths = []
    rank_by_players = []
    for inputs_seq, rank_by_player in batch:
        inputs.append(inputs_seq)
        lengths.append(len(inputs_seq))
        rank_by_players.append(rank_by_player)

    lengths = torch.tensor(lengths)
    rank_by_players = torch.tensor(rank_by_players, dtype=torch.int64, pin_memory=True)

    padded = pad_sequence(inputs, batch_first=True)
    packed_inputs = pack_padded_sequence(
        padded, lengths, batch_first=True, enforce_sorted=False
    )
    packed_inputs.pin_memory()

    return packed_inputs, rank_by_players


def train():
    cfg = config["grp"]
    batch_size = cfg["control"]["batch_size"]
    save_every = cfg["control"]["save_every"]
    val_steps = cfg["control"]["val_steps"]

    device = torch.device(cfg["control"]["device"])
    torch.backends.cudnn.benchmark = cfg["control"]["enable_cudnn_benchmark"]
    if device.type == "cuda":
        logging.info(f"device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        logging.info(f"device: {device}")

    grp = GRP(**cfg["network"]).to(device)
    optimizer = optim.AdamW(grp.parameters())

    state_file = cfg["state_file"]
    if os.path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        logging.info(f"loaded: {timestamp}")
        grp.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        steps = state["steps"]
    else:
        steps = 0

    lr = cfg["optim"]["lr"]
    optimizer.param_groups[0]["lr"] = lr

    # --- Archive-based dataset setup ---
    file_index = cfg["dataset"]["file_index"]
    archive_globs = cfg["dataset"]["archive_globs"]
    val_ratio = cfg["dataset"].get("val_ratio", 0.05)
    log_batch_size = cfg["dataset"].get("log_batch_size", 500)

    if os.path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        train_archive_list = index["train_archive_list"]
        val_archive_list = index["val_archive_list"]
    else:
        logging.info("building archive index...")
        all_archives = []
        for pat in archive_globs:
            all_archives.extend(glob(pat, recursive=True))
        all_archives.sort()

        # Split: last N archives for validation
        n_val = max(1, int(len(all_archives) * val_ratio))
        val_archive_list = all_archives[-n_val:]
        train_archive_list = all_archives[:-n_val]

        torch.save(
            {
                "train_archive_list": train_archive_list,
                "val_archive_list": val_archive_list,
            },
            file_index,
        )

    writer = SummaryWriter(cfg["control"]["tensorboard_dir"])

    logging.info(f"train archives: {len(train_archive_list)}")
    logging.info(f"val archives: {len(val_archive_list)}")
    logging.info(f"total steps: {steps:,}")

    train_data = GrpArchiveDatasetsIter(
        archive_list=train_archive_list,
        log_batch_size=log_batch_size,
        cycle=True,
    )
    train_loader = iter(
        DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,  # archive prefetch is internal, no mp workers
            collate_fn=collate,
        )
    )

    val_data = GrpArchiveDatasetsIter(
        archive_list=val_archive_list,
        log_batch_size=log_batch_size,
        cycle=True,
    )
    val_loader = iter(
        DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
            collate_fn=collate,
        )
    )

    stats = {
        "train_loss": 0,
        "train_acc": 0,
        "val_loss": 0,
        "val_acc": 0,
    }

    pb = tqdm(total=save_every, desc="TRAIN")
    for inputs, rank_by_players in train_loader:
        inputs = inputs.to(dtype=torch.float64, device=device)
        rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)

        logits = grp.forward_packed(inputs)
        labels = grp.get_label(rank_by_players)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            stats["train_loss"] += loss
            stats["train_acc"] += (logits.argmax(-1) == labels).to(torch.float64).mean()

        steps += 1
        pb.update(1)

        if steps % save_every == 0:
            pb.close()

            with torch.inference_mode():
                grp.eval()
                pb = tqdm(total=val_steps, desc="VAL")
                for idx, (inputs, rank_by_players) in enumerate(val_loader):
                    if idx == val_steps:
                        break
                    inputs = inputs.to(dtype=torch.float64, device=device)
                    rank_by_players = rank_by_players.to(
                        dtype=torch.int64, device=device
                    )

                    logits = grp.forward_packed(inputs)
                    labels = grp.get_label(rank_by_players)
                    loss = F.cross_entropy(logits, labels)

                    stats["val_loss"] += loss
                    stats["val_acc"] += (
                        (logits.argmax(-1) == labels).to(torch.float64).mean()
                    )
                    pb.update(1)
                pb.close()
                grp.train()

            writer.add_scalars(
                "loss",
                {
                    "train": stats["train_loss"] / save_every,
                    "val": stats["val_loss"] / val_steps,
                },
                steps,
            )
            writer.add_scalars(
                "acc",
                {
                    "train": stats["train_acc"] / save_every,
                    "val": stats["val_acc"] / val_steps,
                },
                steps,
            )
            writer.add_scalar("lr", lr, steps)
            writer.flush()

            for k in stats:
                stats[k] = 0
            logging.info(f"total steps: {steps:,}")

            state = {
                "model": grp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": steps,
                "timestamp": datetime.now().timestamp(),
            }
            torch.save(state, state_file)
            pb = tqdm(total=save_every, desc="TRAIN")
    pb.close()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        pass
