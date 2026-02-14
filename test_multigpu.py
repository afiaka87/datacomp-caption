"""Tests for multi-GPU captioning support (TDD - written before implementation)."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# We import the functions we'll implement. These don't exist yet, so the
# tests will fail until we implement them.
# ---------------------------------------------------------------------------
from caption_dataset import (
    get_gpu_devices,
    distribute_tars,
    merge_completed_tars,
    save_checkpoint,
    load_checkpoint,
    write_tar_temp,
    merger_loop,
    _get_merged_cursor,
)


# ===========================================================================
# a) GPU auto-detection
# ===========================================================================


class TestGetGpuDevices:
    """Test GPU device detection and selection."""

    @patch("torch.cuda.device_count", return_value=4)
    def test_auto_detect_all_gpus(self, mock_count):
        """With no arguments, should detect and return all available GPUs."""
        devices = get_gpu_devices()
        assert devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    @patch("torch.cuda.device_count", return_value=4)
    def test_num_gpus_limits_count(self, mock_count):
        """--num-gpus N should use the first N GPUs."""
        devices = get_gpu_devices(num_gpus=2)
        assert devices == ["cuda:0", "cuda:1"]

    @patch("torch.cuda.device_count", return_value=2)
    def test_num_gpus_exceeds_available(self, mock_count):
        """Requesting more GPUs than available should raise an error."""
        with pytest.raises(ValueError, match="Requested 4 GPUs but only 2 available"):
            get_gpu_devices(num_gpus=4)

    @patch("torch.cuda.device_count", return_value=4)
    def test_specific_gpu_ids(self, mock_count):
        """--gpus 0,2,3 should use exactly those devices."""
        devices = get_gpu_devices(gpus=[0, 2, 3])
        assert devices == ["cuda:0", "cuda:2", "cuda:3"]

    @patch("torch.cuda.device_count", return_value=2)
    def test_specific_gpu_id_out_of_range(self, mock_count):
        """Requesting a GPU ID that doesn't exist should raise an error."""
        with pytest.raises(ValueError, match="GPU 3 not available"):
            get_gpu_devices(gpus=[0, 3])

    @patch("torch.cuda.device_count", return_value=1)
    def test_single_gpu_auto(self, mock_count):
        """Single GPU system should return just cuda:0."""
        devices = get_gpu_devices()
        assert devices == ["cuda:0"]

    @patch("torch.cuda.device_count", return_value=0)
    def test_no_gpus_available(self, mock_count):
        """No GPUs should raise an error."""
        with pytest.raises(ValueError, match="No CUDA GPUs available"):
            get_gpu_devices()

    @patch("torch.cuda.device_count", return_value=4)
    def test_gpus_overrides_num_gpus(self, mock_count):
        """--gpus should take precedence over --num-gpus."""
        devices = get_gpu_devices(num_gpus=2, gpus=[0, 3])
        assert devices == ["cuda:0", "cuda:3"]


# ===========================================================================
# b) Tar distribution across GPUs
# ===========================================================================


class TestDistributeTars:
    """Test distributing tar files across GPUs in contiguous chunks."""

    def _make_tar_paths(self, n):
        return [Path(f"/data/{i:07d}.tar") for i in range(n)]

    def test_even_split(self):
        """10 tars across 2 GPUs -> 5 each."""
        tars = self._make_tar_paths(10)
        chunks = distribute_tars(tars, num_gpus=2)
        assert len(chunks) == 2
        assert chunks[0] == tars[:5]
        assert chunks[1] == tars[5:]

    def test_uneven_split(self):
        """10 tars across 3 GPUs -> 4, 3, 3."""
        tars = self._make_tar_paths(10)
        chunks = distribute_tars(tars, num_gpus=3)
        assert len(chunks) == 3
        # All tars should be accounted for
        all_assigned = [t for chunk in chunks for t in chunk]
        assert len(all_assigned) == 10
        assert set(all_assigned) == set(tars)
        # Chunks should be contiguous
        assert chunks[0] == tars[:4]
        assert chunks[1] == tars[4:7]
        assert chunks[2] == tars[7:]

    def test_single_gpu(self):
        """1 GPU gets all tars."""
        tars = self._make_tar_paths(10)
        chunks = distribute_tars(tars, num_gpus=1)
        assert len(chunks) == 1
        assert chunks[0] == tars

    def test_more_gpus_than_tars(self):
        """3 tars across 5 GPUs -> 3 non-empty chunks, 2 empty."""
        tars = self._make_tar_paths(3)
        chunks = distribute_tars(tars, num_gpus=5)
        assert len(chunks) == 5
        non_empty = [c for c in chunks if c]
        assert len(non_empty) == 3
        all_assigned = [t for chunk in chunks for t in chunk]
        assert set(all_assigned) == set(tars)

    def test_empty_tar_list(self):
        """No remaining tars should return empty chunks."""
        chunks = distribute_tars([], num_gpus=3)
        assert len(chunks) == 3
        assert all(c == [] for c in chunks)

    def test_contiguous_ordering_preserved(self):
        """Each chunk should be a contiguous slice in sorted order."""
        tars = self._make_tar_paths(100)
        chunks = distribute_tars(tars, num_gpus=4)
        reconstructed = [t for chunk in chunks for t in chunk]
        assert reconstructed == tars


# ===========================================================================
# c) Incremental merging of per-tar JSONL temp files
# ===========================================================================


class TestMergeCompletedTars:
    """Test the incremental merger that combines per-tar temp files in order."""

    def _write_temp_tar_jsonl(self, temp_dir, tar_name, records):
        """Write a temp JSONL file as a worker would."""
        path = temp_dir / f"{tar_name}.jsonl"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        tmp.rename(path)

    def test_merges_in_tar_order(self):
        """Tars should be merged in sorted order, not arrival order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"

            tar_order = ["0000000.tar", "0000001.tar", "0000002.tar"]

            # Write tar 2 first, then tar 0, then tar 1 (out of order)
            self._write_temp_tar_jsonl(
                temp_dir,
                "0000002.tar",
                [
                    {"key": "c1", "tar_file": "0000002.tar"},
                ],
            )
            self._write_temp_tar_jsonl(
                temp_dir,
                "0000000.tar",
                [
                    {"key": "a1", "tar_file": "0000000.tar"},
                    {"key": "a2", "tar_file": "0000000.tar"},
                ],
            )
            self._write_temp_tar_jsonl(
                temp_dir,
                "0000001.tar",
                [
                    {"key": "b1", "tar_file": "0000001.tar"},
                ],
            )

            # Merge
            merged_tars, total_images = merge_completed_tars(
                tar_order=tar_order,
                temp_dir=temp_dir,
                output_path=output_path,
                already_merged=set(),
            )

            # All 3 should be merged
            assert merged_tars == {"0000000.tar", "0000001.tar", "0000002.tar"}
            assert total_images == 4

            # Check output order
            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert [entry["key"] for entry in lines] == ["a1", "a2", "b1", "c1"]

    def test_stops_at_gap(self):
        """If tar 1 is missing, should merge tar 0 but not tar 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"

            tar_order = ["0000000.tar", "0000001.tar", "0000002.tar"]

            self._write_temp_tar_jsonl(
                temp_dir,
                "0000000.tar",
                [
                    {"key": "a1", "tar_file": "0000000.tar"},
                ],
            )
            # Skip tar 1
            self._write_temp_tar_jsonl(
                temp_dir,
                "0000002.tar",
                [
                    {"key": "c1", "tar_file": "0000002.tar"},
                ],
            )

            merged_tars, total_images = merge_completed_tars(
                tar_order=tar_order,
                temp_dir=temp_dir,
                output_path=output_path,
                already_merged=set(),
            )

            # Only tar 0 should be merged
            assert merged_tars == {"0000000.tar"}
            assert total_images == 1

            # Tar 2 temp file should still exist
            assert (temp_dir / "0000002.tar.jsonl").exists()

    def test_resumes_from_already_merged(self):
        """Should skip already-merged tars and continue from the cursor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"

            tar_order = ["0000000.tar", "0000001.tar", "0000002.tar"]

            # Simulate: tar 0 already merged in previous run
            with open(output_path, "w") as f:
                f.write(json.dumps({"key": "a1", "tar_file": "0000000.tar"}) + "\n")

            # New temp files for tar 1 and 2
            self._write_temp_tar_jsonl(
                temp_dir,
                "0000001.tar",
                [
                    {"key": "b1", "tar_file": "0000001.tar"},
                ],
            )
            self._write_temp_tar_jsonl(
                temp_dir,
                "0000002.tar",
                [
                    {"key": "c1", "tar_file": "0000002.tar"},
                ],
            )

            merged_tars, total_images = merge_completed_tars(
                tar_order=tar_order,
                temp_dir=temp_dir,
                output_path=output_path,
                already_merged={"0000000.tar"},
            )

            assert merged_tars == {"0000001.tar", "0000002.tar"}
            assert total_images == 2

            # Output should have all 3 records (1 pre-existing + 2 new)
            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 3
            assert [entry["key"] for entry in lines] == ["a1", "b1", "c1"]

    def test_no_temp_files_available(self):
        """If no temp files exist, nothing should be merged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"

            tar_order = ["0000000.tar", "0000001.tar"]

            merged_tars, total_images = merge_completed_tars(
                tar_order=tar_order,
                temp_dir=temp_dir,
                output_path=output_path,
                already_merged=set(),
            )

            assert merged_tars == set()
            assert total_images == 0
            assert not output_path.exists()

    def test_deletes_temp_files_after_merge(self):
        """Merged temp files should be cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"

            tar_order = ["0000000.tar"]
            self._write_temp_tar_jsonl(
                temp_dir,
                "0000000.tar",
                [
                    {"key": "a1", "tar_file": "0000000.tar"},
                ],
            )

            merge_completed_tars(
                tar_order=tar_order,
                temp_dir=temp_dir,
                output_path=output_path,
                already_merged=set(),
            )

            assert not (temp_dir / "0000000.tar.jsonl").exists()


# ===========================================================================
# d) Resume with different GPU count
# ===========================================================================


class TestResumeWithDifferentGpuCount:
    """Test that resuming with a different number of GPUs works correctly."""

    def _make_tar_paths(self, n):
        return [Path(f"/data/{i:07d}.tar") for i in range(n)]

    def test_resume_fewer_gpus(self):
        """Started with 4 GPUs, 50/100 tars done, resume with 2 GPUs."""
        all_tars = self._make_tar_paths(100)
        completed = {f"{i:07d}.tar" for i in range(50)}
        remaining = [t for t in all_tars if t.name not in completed]

        chunks = distribute_tars(remaining, num_gpus=2)

        assert len(chunks) == 2
        all_assigned = [t for c in chunks for t in c]
        assert len(all_assigned) == 50
        assert set(all_assigned) == set(remaining)

    def test_resume_more_gpus(self):
        """Started with 2 GPUs, 50/100 done, resume with 8 GPUs."""
        all_tars = self._make_tar_paths(100)
        completed = {f"{i:07d}.tar" for i in range(50)}
        remaining = [t for t in all_tars if t.name not in completed]

        chunks = distribute_tars(remaining, num_gpus=8)

        assert len(chunks) == 8
        all_assigned = [t for c in chunks for t in c]
        assert len(all_assigned) == 50
        assert set(all_assigned) == set(remaining)

    def test_resume_single_to_multi(self):
        """Started single-GPU, resume with 4 GPUs. Checkpoint should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "progress.json"

            # Simulate single-GPU checkpoint
            save_checkpoint(
                checkpoint_path,
                {
                    "completed_tars": [f"{i:07d}.tar" for i in range(10)],
                    "merged_up_to": "0000009.tar",
                    "total_images": 100000,
                    "errors": 0,
                },
            )

            # Load and verify
            cp = load_checkpoint(checkpoint_path)
            completed = set(cp["completed_tars"])
            assert len(completed) == 10

            # Distribute remaining across 4 GPUs
            all_tars = self._make_tar_paths(20)
            remaining = [t for t in all_tars if t.name not in completed]
            chunks = distribute_tars(remaining, num_gpus=4)

            all_assigned = [t for c in chunks for t in c]
            assert len(all_assigned) == 10
            # None of the completed tars should be assigned
            for t in all_assigned:
                assert t.name not in completed

    def test_resume_all_done(self):
        """If all tars are done, each GPU should get an empty chunk."""
        all_tars = self._make_tar_paths(10)
        completed = {t.name for t in all_tars}
        remaining = [t for t in all_tars if t.name not in completed]

        chunks = distribute_tars(remaining, num_gpus=4)
        assert all(c == [] for c in chunks)


# ===========================================================================
# e) Checkpoint round-trip with multi-GPU fields
# ===========================================================================


class TestCheckpointMultiGpu:
    """Test checkpoint save/load with multi-GPU-specific fields."""

    def test_checkpoint_round_trip(self):
        """Save and load checkpoint with merged_up_to field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "progress.json"

            state = {
                "completed_tars": ["0000000.tar", "0000001.tar", "0000005.tar"],
                "merged_up_to": "0000001.tar",
                "total_images": 30000,
                "errors": 2,
                "start_time": "2026-02-14T10:00:00",
            }
            save_checkpoint(cp_path, state)
            loaded = load_checkpoint(cp_path)

            assert loaded["completed_tars"] == state["completed_tars"]
            assert loaded["merged_up_to"] == "0000001.tar"
            assert loaded["total_images"] == 30000
            assert loaded["errors"] == 2

    def test_checkpoint_without_merged_up_to(self):
        """Old single-GPU checkpoints won't have merged_up_to - should handle gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "progress.json"

            # Old-style checkpoint (no merged_up_to)
            state = {
                "completed_tars": ["0000000.tar", "0000001.tar"],
                "total_images": 20000,
                "errors": 0,
            }
            save_checkpoint(cp_path, state)
            loaded = load_checkpoint(cp_path)

            # Should not crash when accessing merged_up_to
            assert loaded.get("merged_up_to") is None


# ===========================================================================
# f) write_tar_temp atomicity
# ===========================================================================


class TestWriteTarTemp:
    """Test atomic temp file writing."""

    def test_writes_valid_jsonl(self):
        """Should produce valid JSONL readable back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            records = [
                {"key": "img001", "caption": "A cat"},
                {"key": "img002", "caption": "A dog"},
            ]
            write_tar_temp(temp_dir, "0000000.tar", records)

            path = temp_dir / "0000000.tar.jsonl"
            assert path.exists()

            with open(path) as f:
                loaded = [json.loads(line) for line in f]
            assert loaded == records

    def test_no_leftover_tmp_file(self):
        """The .tmp file should be renamed away, not left behind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            write_tar_temp(temp_dir, "test.tar", [{"key": "x"}])
            assert not (temp_dir / "test.tar.tmp").exists()
            assert (temp_dir / "test.tar.jsonl").exists()


# ===========================================================================
# g) Merged cursor tracking
# ===========================================================================


class TestGetMergedCursor:
    """Test the merge cursor helper."""

    def test_all_merged(self):
        order = ["a.tar", "b.tar", "c.tar"]
        assert _get_merged_cursor(order, {"a.tar", "b.tar", "c.tar"}) == "c.tar"

    def test_partial_merge(self):
        order = ["a.tar", "b.tar", "c.tar"]
        assert _get_merged_cursor(order, {"a.tar", "b.tar"}) == "b.tar"

    def test_gap_in_middle(self):
        """If a.tar and c.tar are merged but b.tar is not, cursor should be a.tar."""
        order = ["a.tar", "b.tar", "c.tar"]
        assert _get_merged_cursor(order, {"a.tar", "c.tar"}) == "a.tar"

    def test_none_merged(self):
        order = ["a.tar", "b.tar"]
        assert _get_merged_cursor(order, set()) is None


# ===========================================================================
# h) Merger loop integration test
# ===========================================================================


class TestMergerLoop:
    """Test the background merger loop with simulated worker output."""

    def test_merger_picks_up_files_and_exits(self):
        """Merger should merge files as they appear, then exit on shutdown."""
        import multiprocessing as mp_test

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"
            checkpoint_path = Path(tmpdir) / "progress.json"

            tar_order = ["0000000.tar", "0000001.tar", "0000002.tar"]
            shutdown_event = mp_test.Event()

            # Pre-write all temp files (simulating workers already done)
            for i, tar_name in enumerate(tar_order):
                write_tar_temp(
                    temp_dir,
                    tar_name,
                    [
                        {"key": f"img{i}", "tar_file": tar_name},
                    ],
                )

            # Run merger in a thread
            merger = threading.Thread(
                target=merger_loop,
                args=(
                    tar_order,
                    temp_dir,
                    output_path,
                    checkpoint_path,
                    set(),
                    shutdown_event,
                    0.1,
                ),
            )
            merger.start()
            merger.join(timeout=10)
            assert not merger.is_alive(), "Merger should have exited"

            # Check output
            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 3
            assert [entry["key"] for entry in lines] == ["img0", "img1", "img2"]

            # Check checkpoint
            cp = load_checkpoint(checkpoint_path)
            assert set(cp["completed_tars"]) == set(tar_order)
            assert cp["merged_up_to"] == "0000002.tar"

    def test_merger_waits_for_gap_then_continues(self):
        """Merger waits when there's a gap, merges when it's filled."""
        import multiprocessing as mp_test

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"
            checkpoint_path = Path(tmpdir) / "progress.json"

            tar_order = ["0000000.tar", "0000001.tar", "0000002.tar"]
            shutdown_event = mp_test.Event()

            # Write tar 0 and 2 (gap at 1)
            write_tar_temp(
                temp_dir, "0000000.tar", [{"key": "a", "tar_file": "0000000.tar"}]
            )
            write_tar_temp(
                temp_dir, "0000002.tar", [{"key": "c", "tar_file": "0000002.tar"}]
            )

            merger = threading.Thread(
                target=merger_loop,
                args=(
                    tar_order,
                    temp_dir,
                    output_path,
                    checkpoint_path,
                    set(),
                    shutdown_event,
                    0.1,
                ),
            )
            merger.start()

            # Give merger time to process tar 0 and stall on tar 1
            time.sleep(0.5)

            # Now fill the gap
            write_tar_temp(
                temp_dir, "0000001.tar", [{"key": "b", "tar_file": "0000001.tar"}]
            )

            # Wait for merger to finish (all tars present)
            merger.join(timeout=10)
            assert not merger.is_alive()

            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert [entry["key"] for entry in lines] == ["a", "b", "c"]
