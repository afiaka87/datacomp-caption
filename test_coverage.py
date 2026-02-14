"""Tests for full coverage of caption_dataset.py."""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import signal
import tarfile
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import caption_dataset as cd


# ===========================================================================
# Helpers
# ===========================================================================


def _make_image(width=8, height=8) -> Image.Image:
    """Create a tiny valid RGB image."""
    return Image.new("RGB", (width, height), color=(255, 0, 0))


def _image_bytes(fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    _make_image().save(buf, format=fmt)
    return buf.getvalue()


def _make_tar(path: Path, samples: list[dict]):
    """Create a tar file with given samples.

    Each sample dict should have 'key' and optionally 'image_bytes', 'caption', 'meta'.
    """
    with tarfile.open(path, "w") as tf:
        for s in samples:
            key = s["key"]
            if "image_bytes" in s:
                data = s["image_bytes"]
                info = tarfile.TarInfo(name=f"{key}.jpg")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            if "caption" in s:
                data = s["caption"].encode()
                info = tarfile.TarInfo(name=f"{key}.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            if "meta" in s:
                data = json.dumps(s["meta"]).encode()
                info = tarfile.TarInfo(name=f"{key}.json")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


# ===========================================================================
# is_shutdown / signal handler
# ===========================================================================


class TestShutdown:
    def setup_method(self):
        cd._shutdown_requested = False

    def teardown_method(self):
        cd._shutdown_requested = False
        cd._shutdown_event = None

    def test_is_shutdown_global_false(self):
        assert cd.is_shutdown() is False

    def test_is_shutdown_global_true(self):
        cd._shutdown_requested = True
        assert cd.is_shutdown() is True

    def test_is_shutdown_event_not_set(self):
        event = mp.Event()
        assert cd.is_shutdown(event) is False

    def test_is_shutdown_event_set(self):
        event = mp.Event()
        event.set()
        assert cd.is_shutdown(event) is True

    def test_signal_handler_first_call(self):
        cd._signal_handler(signal.SIGINT, None)
        assert cd._shutdown_requested is True

    def test_signal_handler_with_event(self):
        event = mp.Event()
        cd._shutdown_event = event
        cd._signal_handler(signal.SIGINT, None)
        assert event.is_set()

    def test_signal_handler_second_call_exits(self):
        cd._shutdown_requested = True
        with pytest.raises(SystemExit):
            cd._signal_handler(signal.SIGINT, None)


# ===========================================================================
# iter_tar_samples
# ===========================================================================


class TestIterTarSamples:
    def test_basic_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            img_data = _image_bytes()
            _make_tar(
                tar_path,
                [
                    {
                        "key": "img001",
                        "image_bytes": img_data,
                        "caption": "A cat",
                        "meta": {"uid": "u1"},
                    },
                    {
                        "key": "img002",
                        "image_bytes": img_data,
                        "caption": "A dog",
                        "meta": {"uid": "u2"},
                    },
                ],
            )

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert len(samples) == 2
            assert samples[0][0] == "img001"
            assert samples[0][2] == "A cat"
            assert samples[0][3] == {"uid": "u1"}
            assert samples[1][0] == "img002"

    def test_missing_image_skips_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": "img001", "caption": "No image here"},  # no image_bytes
                    {
                        "key": "img002",
                        "image_bytes": _image_bytes(),
                        "caption": "Has image",
                    },
                ],
            )

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert len(samples) == 1
            assert samples[0][0] == "img002"

    def test_missing_caption_defaults_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": "img001", "image_bytes": _image_bytes()},  # no caption
                ],
            )

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert len(samples) == 1
            assert samples[0][2] == ""  # default empty caption

    def test_missing_meta_defaults_empty_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": "img001", "image_bytes": _image_bytes()},
                ],
            )

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert samples[0][3] == {}

    def test_no_extension_skipped(self):
        """Files without extensions should be ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            with tarfile.open(tar_path, "w") as tf:
                # File with no extension
                data = b"noext"
                info = tarfile.TarInfo(name="noext")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
                # Valid sample
                img = _image_bytes()
                info = tarfile.TarInfo(name="img001.jpg")
                info.size = len(img)
                tf.addfile(info, io.BytesIO(img))

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert len(samples) == 1

    def test_webp_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            buf = io.BytesIO()
            _make_image().save(buf, format="WEBP")
            webp_bytes = buf.getvalue()

            with tarfile.open(tar_path, "w") as tf:
                info = tarfile.TarInfo(name="img001.webp")
                info.size = len(webp_bytes)
                tf.addfile(info, io.BytesIO(webp_bytes))

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert len(samples) == 1


# ===========================================================================
# get_tar_files
# ===========================================================================


class TestGetTarFiles:
    def test_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["0000002.tar", "0000000.tar", "0000001.tar"]:
                (Path(tmpdir) / name).touch()
            # non-tar file should be excluded
            (Path(tmpdir) / "readme.txt").touch()

            tars = cd.get_tar_files(Path(tmpdir))
            assert [t.name for t in tars] == [
                "0000000.tar",
                "0000001.tar",
                "0000002.tar",
            ]

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert cd.get_tar_files(Path(tmpdir)) == []


# ===========================================================================
# generate_captions (mocked model)
# ===========================================================================


class TestGenerateCaptions:
    def test_basic_generation(self):
        model = MagicMock()
        processor = MagicMock()

        # Mock processor call returns tensor-like inputs
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        processor.return_value = mock_inputs

        # Mock model.generate returns tensor IDs
        model.generate.return_value = MagicMock()

        # Mock batch_decode
        processor.batch_decode.return_value = ["raw output 1", "raw output 2"]

        # Mock post_process_generation
        task = "<DETAILED_CAPTION>"
        processor.post_process_generation.side_effect = [
            {task: "A detailed caption 1"},
            {task: "A detailed caption 2"},
        ]

        images = [_make_image(), _make_image()]
        captions = cd.generate_captions(model, processor, images, task, device="cpu")

        assert captions == ["A detailed caption 1", "A detailed caption 2"]
        model.generate.assert_called_once()

    def test_pad_token_stripped(self):
        model = MagicMock()
        processor = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        processor.return_value = mock_inputs
        model.generate.return_value = MagicMock()
        processor.batch_decode.return_value = ["text with <pad>"]

        task = "<CAPTION>"
        processor.post_process_generation.return_value = {
            task: "caption with <pad> tokens"
        }

        captions = cd.generate_captions(
            model, processor, [_make_image()], task, device="cpu"
        )
        assert captions == ["caption with  tokens"]


# ===========================================================================
# _flush_batch_to_records
# ===========================================================================


class TestFlushBatchToRecords:
    def _mock_generate(self, captions):
        """Patch generate_captions to return given captions."""
        return patch.object(cd, "generate_captions", return_value=captions)

    def test_appends_records(self):
        records = []
        images = [_make_image(), _make_image()]
        stats = {"total_images": 0, "errors": 0}

        with self._mock_generate(["cap1", "cap2"]):
            n = cd._flush_batch_to_records(
                None,
                None,
                records,
                ["k1", "k2"],
                images,
                ["orig1", "orig2"],
                [{"uid": "u1", "url": "http://a", "width": 100, "height": 200}, {}],
                "test.tar",
                "<CAPTION>",
                "cpu",
                stats,
            )

        assert n == 2
        assert len(records) == 2
        assert records[0]["key"] == "k1"
        assert records[0]["caption"] == "cap1"
        assert records[0]["tar_file"] == "test.tar"
        assert records[0]["uid"] == "u1"
        assert records[1]["uid"] == ""
        assert stats["total_images"] == 2

    def test_oom_splits_batch(self):
        records = []
        stats = {"total_images": 0, "errors": 0}
        images = [_make_image() for _ in range(4)]

        call_count = 0

        def mock_gen(model, processor, imgs, task, device):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise torch.cuda.OutOfMemoryError("OOM")
            return [f"cap{i}" for i in range(len(imgs))]

        import torch

        with patch.object(cd, "generate_captions", side_effect=mock_gen):
            with patch("torch.cuda.empty_cache"):
                n = cd._flush_batch_to_records(
                    None,
                    None,
                    records,
                    ["k1", "k2", "k3", "k4"],
                    images,
                    ["o1", "o2", "o3", "o4"],
                    [{}, {}, {}, {}],
                    "test.tar",
                    "<CAPTION>",
                    "cpu",
                    stats,
                )

        assert n == 4
        assert len(records) == 4

    def test_oom_single_image_skipped(self):
        records = []
        stats = {"total_images": 0, "errors": 0}
        images = [_make_image()]

        import torch

        with patch.object(
            cd, "generate_captions", side_effect=torch.cuda.OutOfMemoryError("OOM")
        ):
            with patch("torch.cuda.empty_cache"):
                n = cd._flush_batch_to_records(
                    None,
                    None,
                    records,
                    ["k1"],
                    images,
                    ["o1"],
                    [{}],
                    "test.tar",
                    "<CAPTION>",
                    "cpu",
                    stats,
                )

        assert n == 0
        assert stats["errors"] == 1

    def test_general_exception(self):
        records = []
        stats = {"total_images": 0, "errors": 0}
        images = [_make_image(), _make_image()]

        with patch.object(cd, "generate_captions", side_effect=RuntimeError("bad")):
            n = cd._flush_batch_to_records(
                None,
                None,
                records,
                ["k1", "k2"],
                images,
                ["o1", "o2"],
                [{}, {}],
                "test.tar",
                "<CAPTION>",
                "cpu",
                stats,
            )

        assert n == 0
        assert stats["errors"] == 2


# ===========================================================================
# _flush_batch (writes to file)
# ===========================================================================


class TestFlushBatch:
    def test_writes_jsonl(self):
        output = io.StringIO()
        stats = {"total_images": 0, "errors": 0}
        images = [_make_image()]

        with patch.object(cd, "generate_captions", return_value=["a cat"]):
            n = cd._flush_batch(
                None,
                None,
                output,
                ["k1"],
                images,
                ["orig"],
                [{"uid": "u1", "url": "http://x", "width": 10, "height": 20}],
                "t.tar",
                "<CAPTION>",
                "cpu",
                stats,
            )

        assert n == 1
        output.seek(0)
        rec = json.loads(output.read().strip())
        assert rec["key"] == "k1"
        assert rec["caption"] == "a cat"

    def test_oom_recovery(self):
        output = io.StringIO()
        stats = {"total_images": 0, "errors": 0}
        images = [_make_image(), _make_image()]

        call_count = 0
        import torch

        def mock_gen(model, processor, imgs, task, device):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise torch.cuda.OutOfMemoryError("OOM")
            return ["cap"] * len(imgs)

        with patch.object(cd, "generate_captions", side_effect=mock_gen):
            with patch("torch.cuda.empty_cache"):
                n = cd._flush_batch(
                    None,
                    None,
                    output,
                    ["k1", "k2"],
                    images,
                    ["o1", "o2"],
                    [{}, {}],
                    "t.tar",
                    "<CAPTION>",
                    "cpu",
                    stats,
                )

        assert n == 2


# ===========================================================================
# process_tar_to_records (integration with mock model)
# ===========================================================================


class TestProcessTarToRecords:
    def test_processes_tar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {
                        "key": "img001",
                        "image_bytes": _image_bytes(),
                        "caption": "hello",
                    },
                    {
                        "key": "img002",
                        "image_bytes": _image_bytes(),
                        "caption": "world",
                    },
                ],
            )

            records = []
            stats = {"total_images": 0, "errors": 0}

            with patch.object(
                cd,
                "generate_captions",
                side_effect=lambda m, p, imgs, t, d: ["cap"] * len(imgs),
            ):
                count = cd.process_tar_to_records(
                    tar_path,
                    None,
                    None,
                    records,
                    batch_size=16,
                    task_prompt="<CAPTION>",
                    device="cpu",
                    stats=stats,
                )

            assert count == 2
            assert len(records) == 2

    def test_respects_shutdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": f"img{i:03d}", "image_bytes": _image_bytes()}
                    for i in range(10)
                ],
            )

            records = []
            stats = {"total_images": 0, "errors": 0}
            event = mp.Event()
            event.set()  # shutdown immediately

            with patch.object(cd, "generate_captions", return_value=["cap"]):
                count = cd.process_tar_to_records(
                    tar_path,
                    None,
                    None,
                    records,
                    batch_size=2,
                    task_prompt="<CAPTION>",
                    device="cpu",
                    stats=stats,
                    shutdown_event=event,
                )

            assert count == 0

    def test_updates_image_bar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": f"img{i:03d}", "image_bytes": _image_bytes()}
                    for i in range(5)
                ],
            )

            records = []
            stats = {"total_images": 0, "errors": 0}
            bar = MagicMock()

            with patch.object(
                cd,
                "generate_captions",
                side_effect=lambda m, p, imgs, t, d: ["cap"] * len(imgs),
            ):
                cd.process_tar_to_records(
                    tar_path,
                    None,
                    None,
                    records,
                    batch_size=2,
                    task_prompt="<CAPTION>",
                    device="cpu",
                    stats=stats,
                    image_bar=bar,
                )

            # Should have called update at least once
            assert bar.update.called


# ===========================================================================
# process_tar (writes to file directly)
# ===========================================================================


class TestProcessTar:
    def test_writes_to_output_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": "img001", "image_bytes": _image_bytes(), "caption": "hi"},
                ],
            )

            output = io.StringIO()
            stats = {"total_images": 0, "errors": 0}

            with patch.object(cd, "generate_captions", return_value=["generated"]):
                count = cd.process_tar(
                    tar_path,
                    None,
                    None,
                    output,
                    batch_size=16,
                    task_prompt="<CAPTION>",
                    device="cpu",
                    stats=stats,
                )

            assert count == 1
            output.seek(0)
            rec = json.loads(output.read().strip())
            assert rec["caption"] == "generated"

    def test_updates_image_bar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            # Need more images than batch_size to trigger set_postfix_str
            _make_tar(
                tar_path,
                [
                    {"key": f"img{i:03d}", "image_bytes": _image_bytes()}
                    for i in range(3)
                ],
            )

            output = io.StringIO()
            stats = {"total_images": 0, "errors": 0}
            bar = MagicMock()

            with patch.object(
                cd,
                "generate_captions",
                side_effect=lambda m, p, imgs, t, d: ["cap"] * len(imgs),
            ):
                cd.process_tar(
                    tar_path,
                    None,
                    None,
                    output,
                    batch_size=2,
                    task_prompt="<CAPTION>",
                    device="cpu",
                    stats=stats,
                    image_bar=bar,
                )

            assert bar.update.called
            assert bar.set_postfix_str.called


# ===========================================================================
# load_model (mocked)
# ===========================================================================


class TestLoadModel:
    @patch("caption_dataset.torch.compile", side_effect=RuntimeError("no compile"))
    @patch("caption_dataset.Florence2ForConditionalGeneration")
    @patch("caption_dataset.AutoProcessor")
    @patch("caption_dataset.torch.cuda.memory_allocated", return_value=500_000_000)
    def test_load_model_compile_fails_gracefully(
        self, mock_mem, mock_proc_cls, mock_model_cls, mock_compile
    ):
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model

        model, processor = cd.load_model("test-model", "cpu")
        assert model is not None
        assert processor is not None

    @patch("caption_dataset.torch.compile", return_value=MagicMock())
    @patch("caption_dataset.Florence2ForConditionalGeneration")
    @patch("caption_dataset.AutoProcessor")
    @patch("caption_dataset.torch.cuda.memory_allocated", return_value=500_000_000)
    def test_load_model_compile_succeeds(
        self, mock_mem, mock_proc_cls, mock_model_cls, mock_compile
    ):
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model

        model, processor = cd.load_model("test-model", "cpu")
        mock_compile.assert_called_once()


# ===========================================================================
# worker_process (mocked)
# ===========================================================================


class TestWorkerProcess:
    @patch.object(cd, "load_model")
    @patch.object(cd, "process_tar_to_records")
    def test_worker_writes_temp_files(self, mock_process, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_process.return_value = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()

            # Create dummy tar paths (don't need real files since process is mocked)
            tar_files = [Path(f"/fake/{i:07d}.tar") for i in range(2)]
            event = mp.Event()

            cd.worker_process(
                gpu_idx=0,
                device="cuda:0",
                tar_files=tar_files,
                model_id="test-model",
                task_prompt="<CAPTION>",
                batch_size=16,
                temp_dir=temp_dir,
                shutdown_event=event,
                num_gpus=1,
            )

            # Should have written 2 temp files
            assert (temp_dir / "0000000.tar.jsonl").exists()
            assert (temp_dir / "0000001.tar.jsonl").exists()

    @patch.object(cd, "load_model")
    @patch.object(cd, "process_tar_to_records")
    def test_worker_stops_on_shutdown(self, mock_process, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_process.return_value = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()

            tar_files = [Path(f"/fake/{i:07d}.tar") for i in range(10)]
            event = mp.Event()
            event.set()  # pre-set shutdown

            cd.worker_process(
                gpu_idx=0,
                device="cuda:0",
                tar_files=tar_files,
                model_id="test-model",
                task_prompt="<CAPTION>",
                batch_size=16,
                temp_dir=temp_dir,
                shutdown_event=event,
                num_gpus=1,
            )

            # Should not have processed any tars
            assert not list(temp_dir.glob("*.jsonl"))


# ===========================================================================
# Overwrite guard in main()
# ===========================================================================


class TestOverwriteGuard:
    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0"])
    @patch("caption_dataset.get_tar_files", return_value=[])
    def test_existing_output_prompts_quit(self, mock_tars, mock_gpus):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            (output_dir / "datacomp-10m-captions.jsonl").write_text('{"key":"x"}\n')

            with patch(
                "sys.argv",
                ["prog", "--data-dir", tmpdir, "--output-dir", str(output_dir)],
            ):
                with patch("builtins.input", return_value="q"):
                    cd.main()

            # File should still exist, untouched
            assert (
                output_dir / "datacomp-10m-captions.jsonl"
            ).read_text() == '{"key":"x"}\n'

    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0"])
    @patch("caption_dataset.get_tar_files", return_value=[])
    def test_existing_output_prompts_resume(self, mock_tars, mock_gpus):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            (output_dir / "datacomp-10m-captions.jsonl").write_text('{"key":"x"}\n')
            cd.save_checkpoint(
                output_dir / "progress.json",
                {
                    "completed_tars": ["0000000.tar"],
                    "merged_up_to": "0000000.tar",
                    "total_images": 1,
                },
            )

            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with patch(
                "sys.argv",
                ["prog", "--data-dir", str(data_dir), "--output-dir", str(output_dir)],
            ):
                with patch("builtins.input", return_value="r"):
                    with patch.object(cd, "get_tar_files", return_value=[]):
                        cd.main()  # should proceed to "All tars already completed!"

    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0"])
    def test_no_guard_when_resume_flag(self, mock_gpus):
        """--resume flag should skip the guard entirely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            (output_dir / "datacomp-10m-captions.jsonl").write_text('{"key":"x"}\n')

            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with patch(
                "sys.argv",
                [
                    "prog",
                    "--resume",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                with patch.object(cd, "get_tar_files", return_value=[]):
                    with patch("builtins.input") as mock_input:
                        cd.main()
                        mock_input.assert_not_called()

    def test_list_models(self, capsys):
        """--list-models should print models and exit."""
        with patch("sys.argv", ["prog", "--list-models"]):
            cd.main()
        output = capsys.readouterr().out
        assert "Florence-2-base" in output
        assert "Florence-2-large" in output

    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0"])
    def test_no_guard_when_no_existing_files(self, mock_gpus):
        """Fresh start with no existing output should not prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with patch("sys.argv", ["prog", "--data-dir", str(data_dir)]):
                with patch.object(cd, "get_tar_files", return_value=[]):
                    with patch("builtins.input") as mock_input:
                        cd.main()
                        mock_input.assert_not_called()


# ===========================================================================
# write_tar_temp
# ===========================================================================


class TestWriteTarTempUnicode:
    def test_unicode_in_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            records = [{"key": "img001", "caption": "Café résumé 日本語"}]
            cd.write_tar_temp(temp_dir, "test.tar", records)

            with open(temp_dir / "test.tar.jsonl") as f:
                loaded = json.loads(f.readline())
            assert loaded["caption"] == "Café résumé 日本語"


# ===========================================================================
# merger_loop shutdown path
# ===========================================================================


# ===========================================================================
# iter_tar_samples: corrupt txt and json exception branches (lines 236-237, 241-242)
# ===========================================================================


class TestIterTarSamplesCorruptTextAndJson:
    def test_corrupt_txt_skipped(self):
        """A .txt member whose data cannot be decoded should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            img_data = _image_bytes()
            with tarfile.open(tar_path, "w") as tf:
                info = tarfile.TarInfo(name="img001.jpg")
                info.size = len(img_data)
                tf.addfile(info, io.BytesIO(img_data))
                # Create a txt member that is actually a directory type — extractfile returns None
                info = tarfile.TarInfo(name="img001.txt")
                info.size = 0
                # Make it a regular file but with 0 size — extractfile returns empty BytesIO
                # Instead, use a mock to force the exception
                tf.addfile(info, io.BytesIO(b""))

            # Patch to make .txt extraction raise
            orig_extractfile = tarfile.TarFile.extractfile

            def patched_extractfile(self, member):
                if hasattr(member, "name") and member.name.endswith(".txt"):
                    raise OSError("simulated read error")
                return orig_extractfile(self, member)

            with patch.object(tarfile.TarFile, "extractfile", patched_extractfile):
                samples = list(cd.iter_tar_samples(str(tar_path)))

            assert len(samples) == 1
            assert samples[0][2] == ""  # caption defaults empty

    def test_corrupt_json_skipped(self):
        """A .json that can't be parsed should not crash, meta defaults to {}."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            img_data = _image_bytes()
            with tarfile.open(tar_path, "w") as tf:
                info = tarfile.TarInfo(name="img001.jpg")
                info.size = len(img_data)
                tf.addfile(info, io.BytesIO(img_data))
                # Invalid JSON
                bad_json = b"{not valid json"
                info = tarfile.TarInfo(name="img001.json")
                info.size = len(bad_json)
                tf.addfile(info, io.BytesIO(bad_json))

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert len(samples) == 1
            assert samples[0][3] == {}  # meta defaults empty


# ===========================================================================
# _flush_batch: OOM single image + general exception (lines 374-376, 382-385)
# ===========================================================================


class TestFlushBatchOomAndException:
    def test_oom_single_image(self):
        import torch

        output = io.StringIO()
        stats = {"total_images": 0, "errors": 0}
        images = [_make_image()]

        with patch.object(
            cd, "generate_captions", side_effect=torch.cuda.OutOfMemoryError("OOM")
        ):
            with patch("torch.cuda.empty_cache"):
                n = cd._flush_batch(
                    None,
                    None,
                    output,
                    ["k1"],
                    images,
                    ["o1"],
                    [{}],
                    "t.tar",
                    "<CAPTION>",
                    "cpu",
                    stats,
                )

        assert n == 0
        assert stats["errors"] == 1

    def test_general_exception(self):
        output = io.StringIO()
        stats = {"total_images": 0, "errors": 0}
        images = [_make_image(), _make_image()]

        with patch.object(cd, "generate_captions", side_effect=RuntimeError("bad")):
            n = cd._flush_batch(
                None,
                None,
                output,
                ["k1", "k2"],
                images,
                ["o1", "o2"],
                [{}, {}],
                "t.tar",
                "<CAPTION>",
                "cpu",
                stats,
            )

        assert n == 0
        assert stats["errors"] == 2


# ===========================================================================
# worker_process: shutdown DURING process_tar_to_records (lines 473-475)
# ===========================================================================


class TestWorkerShutdownDuringTar:
    @patch.object(cd, "load_model")
    def test_shutdown_during_processing_discards_partial(self, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()

            tar_files = [Path(f"/fake/{i:07d}.tar") for i in range(3)]
            event = mp.Event()

            call_count = 0

            def mock_process(
                tar_path, model, proc, records, bs, tp, dev, stats, se, image_bar=None
            ):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    # Simulate shutdown happening DURING processing of 2nd tar
                    event.set()
                return 5

            with patch.object(cd, "process_tar_to_records", side_effect=mock_process):
                cd.worker_process(
                    gpu_idx=0,
                    device="cuda:0",
                    tar_files=tar_files,
                    model_id="test",
                    task_prompt="<CAPTION>",
                    batch_size=16,
                    temp_dir=temp_dir,
                    shutdown_event=event,
                    num_gpus=1,
                )

            # Only the first tar should have a temp file; second was mid-processing when shutdown hit
            assert (temp_dir / "0000000.tar.jsonl").exists()
            assert not (temp_dir / "0000001.tar.jsonl").exists()
            assert not (temp_dir / "0000002.tar.jsonl").exists()


# ===========================================================================
# merger_loop: shutdown final merge WITH newly_merged (line 633)
# ===========================================================================


class TestMergerLoopShutdownWithNewlyMerged:
    def test_shutdown_final_merge_saves_checkpoint(self):
        """Shutdown final pass finds new file -> saves checkpoint (line 633)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"
            checkpoint_path = Path(tmpdir) / "progress.json"

            tar_order = ["0000000.tar", "0000001.tar"]
            shutdown_event = mp.Event()

            # Control merge_completed_tars to make the file "appear" only on the final pass
            call_count = 0
            orig_merge = cd.merge_completed_tars

            def controlled_merge(tar_order, temp_dir, output_path, already_merged):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call in main loop: nothing to merge, but set shutdown
                    shutdown_event.set()
                    return set(), 0
                # Second call (final pass after shutdown): write and merge tar 0
                cd.write_tar_temp(
                    temp_dir, "0000000.tar", [{"key": "a", "tar_file": "0000000.tar"}]
                )
                return orig_merge(tar_order, temp_dir, output_path, already_merged)

            with patch.object(cd, "merge_completed_tars", side_effect=controlled_merge):
                cd.merger_loop(
                    tar_order,
                    temp_dir,
                    output_path,
                    checkpoint_path,
                    set(),
                    shutdown_event,
                    poll_interval=0.1,
                )

            # Checkpoint should have been saved in the shutdown final merge path (line 633)
            cp = cd.load_checkpoint(checkpoint_path)
            assert "0000000.tar" in cp["completed_tars"]


class TestMergerLoopShutdown:
    def test_shutdown_during_gap_does_final_merge(self):
        """When shutdown fires while waiting at a gap, merger should do final pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"
            checkpoint_path = Path(tmpdir) / "progress.json"

            tar_order = ["0000000.tar", "0000001.tar"]
            shutdown_event = mp.Event()

            # Only tar 0 available
            cd.write_tar_temp(
                temp_dir, "0000000.tar", [{"key": "a", "tar_file": "0000000.tar"}]
            )

            import threading

            def delayed_shutdown():
                time.sleep(0.3)
                shutdown_event.set()

            threading.Thread(target=delayed_shutdown, daemon=True).start()

            cd.merger_loop(
                tar_order,
                temp_dir,
                output_path,
                checkpoint_path,
                set(),
                shutdown_event,
                poll_interval=0.1,
            )

            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 1
            assert lines[0]["key"] == "a"

    def test_shutdown_final_merge_with_new_files(self):
        """Final merge pass on shutdown should pick up files that arrived during gap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "temp"
            temp_dir.mkdir()
            output_path = Path(tmpdir) / "output.jsonl"
            checkpoint_path = Path(tmpdir) / "progress.json"

            tar_order = ["0000000.tar", "0000001.tar"]
            shutdown_event = mp.Event()

            # Write tar 0 initially
            cd.write_tar_temp(
                temp_dir, "0000000.tar", [{"key": "a", "tar_file": "0000000.tar"}]
            )

            import threading

            def delayed_fill_and_shutdown():
                time.sleep(0.3)
                # Fill the gap and immediately shutdown
                cd.write_tar_temp(
                    temp_dir, "0000001.tar", [{"key": "b", "tar_file": "0000001.tar"}]
                )
                shutdown_event.set()

            threading.Thread(target=delayed_fill_and_shutdown, daemon=True).start()

            cd.merger_loop(
                tar_order,
                temp_dir,
                output_path,
                checkpoint_path,
                set(),
                shutdown_event,
                poll_interval=0.1,
            )

            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert [entry["key"] for entry in lines] == ["a", "b"]


# ===========================================================================
# iter_tar_samples exception branches
# ===========================================================================


class TestIterTarSamplesExceptions:
    def test_corrupt_image_skipped(self):
        """Corrupt image data should be skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            with tarfile.open(tar_path, "w") as tf:
                # Corrupt image
                bad_data = b"not a real image"
                info = tarfile.TarInfo(name="img001.jpg")
                info.size = len(bad_data)
                tf.addfile(info, io.BytesIO(bad_data))
                # Good image following
                good_data = _image_bytes()
                info = tarfile.TarInfo(name="img002.jpg")
                info.size = len(good_data)
                tf.addfile(info, io.BytesIO(good_data))

            samples = list(cd.iter_tar_samples(str(tar_path)))
            # img001 has corrupt image so is skipped, img002 should work
            assert len(samples) == 1
            assert samples[0][0] == "img002"

    def test_extractfile_returns_none(self):
        """When extractfile returns None for jpg/txt/json, entries should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            img_data = _image_bytes()
            _make_tar(
                tar_path,
                [
                    {
                        "key": "img001",
                        "image_bytes": img_data,
                        "caption": "hello",
                        "meta": {"uid": "u1"},
                    },
                ],
            )

            def patched_extractfile(self, member):
                return None

            with patch.object(tarfile.TarFile, "extractfile", patched_extractfile):
                samples = list(cd.iter_tar_samples(str(tar_path)))

            # No valid image means no sample yielded
            assert len(samples) == 0

    def test_directory_entries_skipped(self):
        """Directory entries in tar should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            with tarfile.open(tar_path, "w") as tf:
                # Add a directory entry
                dir_info = tarfile.TarInfo(name="subdir")
                dir_info.type = tarfile.DIRTYPE
                tf.addfile(dir_info)
                # Add valid image
                img_data = _image_bytes()
                info = tarfile.TarInfo(name="img001.jpg")
                info.size = len(img_data)
                tf.addfile(info, io.BytesIO(img_data))

            samples = list(cd.iter_tar_samples(str(tar_path)))
            assert len(samples) == 1


# ===========================================================================
# process_tar shutdown mid-batch and remainder flush
# ===========================================================================


class TestProcessTarShutdown:
    def test_shutdown_mid_batch(self):
        """Should stop iterating when shutdown is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": f"img{i:03d}", "image_bytes": _image_bytes()}
                    for i in range(10)
                ],
            )

            output = io.StringIO()
            stats = {"total_images": 0, "errors": 0}
            event = mp.Event()
            event.set()

            with patch.object(cd, "generate_captions", return_value=["cap"]):
                count = cd.process_tar(
                    tar_path,
                    None,
                    None,
                    output,
                    batch_size=2,
                    task_prompt="<CAPTION>",
                    device="cpu",
                    stats=stats,
                    shutdown_event=event,
                )

            assert count == 0

    def test_remainder_flush(self):
        """When images don't fill a full batch, remainder should still be flushed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "test.tar"
            _make_tar(
                tar_path,
                [
                    {"key": f"img{i:03d}", "image_bytes": _image_bytes()}
                    for i in range(3)
                ],
            )

            output = io.StringIO()
            stats = {"total_images": 0, "errors": 0}

            with patch.object(
                cd,
                "generate_captions",
                side_effect=lambda m, p, imgs, t, d: ["cap"] * len(imgs),
            ):
                count = cd.process_tar(
                    tar_path,
                    None,
                    None,
                    output,
                    batch_size=100,
                    task_prompt="<CAPTION>",
                    device="cpu",
                    stats=stats,
                )

            assert count == 3


# ===========================================================================
# _run_single_gpu integration test (mocked model)
# ===========================================================================


class TestRunSingleGpu:
    def test_processes_and_merges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            temp_dir = output_dir / "temp"
            temp_dir.mkdir()

            # Create 2 small tar files
            for i in range(2):
                _make_tar(
                    data_dir / f"{i:07d}.tar",
                    [
                        {
                            "key": f"img{i}_{j}",
                            "image_bytes": _image_bytes(),
                            "caption": f"orig{j}",
                            "meta": {
                                "uid": f"u{j}",
                                "url": "http://x",
                                "width": 10,
                                "height": 10,
                            },
                        }
                        for j in range(3)
                    ],
                )

            tar_files = cd.get_tar_files(data_dir)
            tar_order = [t.name for t in tar_files]
            output_path = output_dir / "output.jsonl"
            checkpoint_path = output_dir / "progress.json"
            stats_path = output_dir / "stats.json"

            args = MagicMock()
            args.model = "test-model"
            args.detail = "detailed"
            args.batch_size = 16

            with patch.object(
                cd, "load_model", return_value=(MagicMock(), MagicMock())
            ):
                with patch.object(
                    cd,
                    "generate_captions",
                    side_effect=lambda m, p, imgs, t, d: [
                        f"gen_{i}" for i in range(len(imgs))
                    ],
                ):
                    cd._run_single_gpu(
                        args,
                        "cpu",
                        tar_files,
                        tar_files,
                        tar_order,
                        set(),
                        set(),
                        "<DETAILED_CAPTION>",
                        output_path,
                        checkpoint_path,
                        stats_path,
                        temp_dir,
                    )

            # Check output JSONL
            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 6  # 2 tars * 3 images

            # Check checkpoint
            cp = cd.load_checkpoint(checkpoint_path)
            assert len(cp["completed_tars"]) == 2

            # Check stats
            with open(stats_path) as f:
                stats = json.load(f)
            assert stats["finished"] is True

    def test_shutdown_interrupts(self):
        """_run_single_gpu should stop on shutdown and report interrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            temp_dir = output_dir / "temp"
            temp_dir.mkdir()

            _make_tar(
                data_dir / "0000000.tar",
                [
                    {"key": "img0", "image_bytes": _image_bytes()},
                ],
            )

            tar_files = cd.get_tar_files(data_dir)
            tar_order = [t.name for t in tar_files]
            output_path = output_dir / "output.jsonl"
            checkpoint_path = output_dir / "progress.json"
            stats_path = output_dir / "stats.json"

            args = MagicMock()
            args.model = "test-model"
            args.detail = "detailed"
            args.batch_size = 16

            # Pre-set shutdown
            cd._shutdown_requested = True
            try:
                with patch.object(
                    cd, "load_model", return_value=(MagicMock(), MagicMock())
                ):
                    cd._run_single_gpu(
                        args,
                        "cpu",
                        tar_files,
                        tar_files,
                        tar_order,
                        set(),
                        set(),
                        "<DETAILED_CAPTION>",
                        output_path,
                        checkpoint_path,
                        stats_path,
                        temp_dir,
                    )
            finally:
                cd._shutdown_requested = False

            # Should not have processed anything
            assert not output_path.exists() or output_path.stat().st_size == 0


# ===========================================================================
# Multi-GPU main() orchestration (mocked workers)
# ===========================================================================


class TestMainMultiGpu:
    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0", "cuda:1"])
    @patch("caption_dataset.mp.get_context")
    def test_multi_gpu_spawns_workers(self, mock_ctx, mock_gpus):
        """Multi-GPU path should spawn worker processes and merger thread."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            output_dir = Path(tmpdir) / "captions"

            # Create tar files
            for i in range(4):
                (data_dir / f"{i:07d}.tar").touch()

            # Mock the spawned processes
            mock_process = MagicMock()
            mock_ctx.return_value.Process.return_value = mock_process

            with patch(
                "sys.argv",
                ["prog", "--data-dir", str(data_dir), "--output-dir", str(output_dir)],
            ):
                with patch.object(cd, "merger_loop"):
                    with patch("threading.Thread") as mock_thread:
                        mock_thread_inst = MagicMock()
                        mock_thread.return_value = mock_thread_inst

                        cd.main()

            # Should have spawned 2 worker processes
            assert mock_ctx.return_value.Process.call_count == 2
            assert mock_process.start.call_count == 2
            assert mock_process.join.call_count == 2

    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0", "cuda:1"])
    @patch("caption_dataset.mp.get_context")
    def test_multi_gpu_resume_catch_up_merge(self, mock_ctx, mock_gpus):
        """Resume in multi-GPU mode should do catch-up merge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            temp_dir = output_dir / "temp"
            temp_dir.mkdir()

            for i in range(4):
                (data_dir / f"{i:07d}.tar").touch()

            # Simulate: tar 0 completed but not merged
            cd.save_checkpoint(
                output_dir / "progress.json",
                {
                    "completed_tars": ["0000000.tar"],
                    "total_images": 100,
                },
            )
            cd.write_tar_temp(
                temp_dir, "0000000.tar", [{"key": "a", "tar_file": "0000000.tar"}]
            )

            mock_process = MagicMock()
            mock_ctx.return_value.Process.return_value = mock_process

            with patch(
                "sys.argv",
                [
                    "prog",
                    "--resume",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                with patch("threading.Thread") as mock_thread:
                    mock_thread.return_value = MagicMock()
                    cd.main()

            # Catch-up merge should have processed the temp file
            output_path = output_dir / "datacomp-10m-captions.jsonl"
            assert output_path.exists()
            with open(output_path) as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 1
            assert lines[0]["key"] == "a"

    @patch(
        "caption_dataset.get_gpu_devices", return_value=["cuda:0", "cuda:1", "cuda:2"]
    )
    @patch("caption_dataset.mp.get_context")
    def test_empty_chunk_skipped(self, mock_ctx, mock_gpus):
        """GPU with empty chunk (more GPUs than tars) should not spawn a process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            # Only 2 tars for 3 GPUs — one GPU gets empty chunk
            for i in range(2):
                (data_dir / f"{i:07d}.tar").touch()

            mock_process = MagicMock()
            mock_ctx.return_value.Process.return_value = mock_process

            with patch("sys.argv", ["prog", "--data-dir", str(data_dir)]):
                with patch("threading.Thread") as mock_thread:
                    mock_thread.return_value = MagicMock()
                    cd.main()

            # Should only spawn 2 processes, not 3
            assert mock_ctx.return_value.Process.call_count == 2


# ===========================================================================
# main() single-GPU dispatch (lines 769-775)
# ===========================================================================


class TestMainSingleGpuDispatch:
    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0"])
    @patch.object(cd, "_run_single_gpu")
    def test_single_gpu_calls_run_single_gpu(self, mock_run, mock_gpus):
        """With 1 GPU, main() should dispatch to _run_single_gpu."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "0000000.tar").touch()

            with patch("sys.argv", ["prog", "--data-dir", str(data_dir)]):
                cd.main()

            mock_run.assert_called_once()


# ===========================================================================
# main() resume with merged_up_to cursor (lines 750-753)
# ===========================================================================


class TestMainResumeWithMergedCursor:
    @patch("caption_dataset.get_gpu_devices", return_value=["cuda:0"])
    @patch.object(cd, "_run_single_gpu")
    def test_resume_reconstructs_already_merged(self, mock_run, mock_gpus):
        """Resume should reconstruct already_merged set from merged_up_to cursor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            for i in range(5):
                (data_dir / f"{i:07d}.tar").touch()

            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            (output_dir / "temp").mkdir()

            # 3 tars completed, merged up to tar 1 (tar 2 completed but not merged)
            cd.save_checkpoint(
                output_dir / "progress.json",
                {
                    "completed_tars": ["0000000.tar", "0000001.tar", "0000002.tar"],
                    "merged_up_to": "0000001.tar",
                    "total_images": 30000,
                },
            )

            with patch(
                "sys.argv",
                [
                    "prog",
                    "--resume",
                    "--data-dir",
                    str(data_dir),
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                cd.main()

            # Verify _run_single_gpu was called
            mock_run.assert_called_once()
            # Check that already_merged was reconstructed correctly (tars 0 and 1)
            call_args = mock_run.call_args
            already_merged = call_args[0][6]  # 7th positional arg (index 6)
            assert already_merged == {"0000000.tar", "0000001.tar"}


# ===========================================================================
# _run_single_gpu: shutdown AFTER process_tar_to_records (line 887)
# ===========================================================================


class TestRunSingleGpuShutdownAfterProcessing:
    def test_shutdown_after_tar_processing(self):
        """Shutdown set during process_tar_to_records should break before writing temp file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            output_dir = Path(tmpdir) / "captions"
            output_dir.mkdir()
            temp_dir = output_dir / "temp"
            temp_dir.mkdir()

            _make_tar(
                data_dir / "0000000.tar",
                [
                    {"key": "img0", "image_bytes": _image_bytes()},
                ],
            )
            _make_tar(
                data_dir / "0000001.tar",
                [
                    {"key": "img1", "image_bytes": _image_bytes()},
                ],
            )

            tar_files = cd.get_tar_files(data_dir)
            tar_order = [t.name for t in tar_files]
            output_path = output_dir / "output.jsonl"
            checkpoint_path = output_dir / "progress.json"
            stats_path = output_dir / "stats.json"

            args = MagicMock()
            args.model = "test-model"
            args.detail = "detailed"
            args.batch_size = 16

            call_count = 0

            def mock_process(tp, m, p, recs, bs, prompt, dev, stats, image_bar=None):
                nonlocal call_count
                call_count += 1
                recs.append({"key": f"k{call_count}", "tar_file": tp.name})
                if call_count == 1:
                    # Set shutdown DURING first tar's processing
                    cd._shutdown_requested = True
                return 1

            try:
                with patch.object(
                    cd, "load_model", return_value=(MagicMock(), MagicMock())
                ):
                    with patch.object(
                        cd, "process_tar_to_records", side_effect=mock_process
                    ):
                        cd._run_single_gpu(
                            args,
                            "cpu",
                            tar_files,
                            tar_files,
                            tar_order,
                            set(),
                            set(),
                            "<DETAILED_CAPTION>",
                            output_path,
                            checkpoint_path,
                            stats_path,
                            temp_dir,
                        )
            finally:
                cd._shutdown_requested = False

            # Shutdown was detected after process_tar_to_records returned,
            # so the temp file should NOT be written (line 886-887 breaks before write)
            assert not (temp_dir / "0000000.tar.jsonl").exists()
            assert not (temp_dir / "0000001.tar.jsonl").exists()
            # Only called once — second tar never started
            assert call_count == 1


# ===========================================================================
# __main__ guard (line 938)
# ===========================================================================


class TestMainGuard:
    def test_main_guard(self):
        """The if __name__ == '__main__' block should call main() (line 938)."""
        import runpy

        with patch("sys.argv", ["caption_dataset.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path(
                    str(Path(cd.__file__).resolve()),
                    run_name="__main__",
                )
            assert exc_info.value.code == 0
