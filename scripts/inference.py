#!/usr/bin/env python3
"""
Moonshine ASR Inference Script

Publication-ready inference script for fine-tuned Moonshine models.
Supports batch processing, GPU acceleration, live transcription, ONNX Runtime (super fast), and various output formats.

Usage:
    # Single file (PyTorch)
    python inference.py --model ./results-moonshine-fr-no-curriculum/final --audio sample.wav

    # ONNX Runtime (super fast!)
    python inference.py --model ./model-onnx --audio sample.wav --onnx

    # Manual ONNX (fastest)
    python inference.py --model ./model-onnx --audio sample.wav --use-manual-onnx

    # Directory of files
    python inference.py --model ./results-moonshine-fr-no-curriculum/final --audio ./test_audio/ --output ./transcriptions.json

    # Live transcription from microphone
    python inference.py --model ./model --live

    # With custom generation parameters
    python inference.py --model ./model --audio audio.wav --num-beams 5 --repetition-penalty 1.3
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
import numpy as np
from transformers import (
    MoonshineForConditionalGeneration,
    AutoProcessor
)
from tqdm import tqdm

# Live mode dependencies (optional)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    sd = None

# ONNX Runtime dependencies (optional)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

try:
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    ORTModelForSpeechSeq2Seq = None


def load_audio(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """
    Load and resample audio file.

    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (default: 16000 Hz)

    Returns:
        Audio array at target sampling rate
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    # Convert to numpy and squeeze
    audio_array = waveform.squeeze().numpy()

    return audio_array


def normalize_audio(audio_data: np.ndarray, target_rms: float = 0.075) -> np.ndarray:
    """
    Normalize audio amplitude to match training data.

    Args:
        audio_data: Input audio array
        target_rms: Target RMS level (default: 0.075, matching preprocessing)

    Returns:
        Normalized audio array
    """
    rms = np.sqrt(np.mean(audio_data**2))
    if rms > 0.001:  # Avoid division by very small numbers
        scale_factor = target_rms / rms
        normalized = audio_data * scale_factor
        return np.clip(normalized, -1.0, 1.0)
    return audio_data


def prepare_audio_input(
    audio: Union[np.ndarray, Path, str],
    sampling_rate: int = 16000
) -> tuple[np.ndarray, float]:
    """
    Resolve path/array audio input into a mono float32 waveform and duration.
    """
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        audio_array = load_audio(audio_path, target_sr=sampling_rate)
    else:
        audio_array = np.asarray(audio, dtype=np.float32)
        audio_array = np.squeeze(audio_array)
        if audio_array.ndim > 1:
            channel_axis = 0 if audio_array.shape[0] <= audio_array.shape[-1] else -1
            audio_array = audio_array.mean(axis=channel_axis)

    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_duration = len(audio_array) / sampling_rate
    return audio_array, audio_duration


def transcribe_with_sliding_windows(
    pipeline,
    audio: Union[np.ndarray, Path, str],
    sampling_rate: int = 16000,
    window_seconds: float = 5.0,
    stride_seconds: Optional[float] = None,
    min_window_seconds: float = 0.5,
    **kwargs
) -> Dict:
    """
    Transcribe long audio by running simple fixed-size windows and concatenating text.

    This is intentionally simple and does not deduplicate overlap. It is useful for
    quickly checking whether full-context decoding is the source of hallucinations.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")

    if stride_seconds is None:
        stride_seconds = window_seconds

    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")

    audio_array, audio_duration = prepare_audio_input(audio, sampling_rate=sampling_rate)
    total_samples = len(audio_array)
    if total_samples == 0:
        raise ValueError("Audio is empty")

    window_samples = max(1, int(round(window_seconds * sampling_rate)))
    stride_samples = max(1, int(round(stride_seconds * sampling_rate)))

    if total_samples <= window_samples:
        start_samples = [0]
    else:
        max_start = total_samples - window_samples
        start_samples = list(range(0, max_start + 1, stride_samples))
        if start_samples[-1] != max_start:
            start_samples.append(max_start)

    segments = []
    combined_text_parts = []
    total_inference_time = 0.0

    for segment_index, start_sample in enumerate(start_samples):
        end_sample = min(start_sample + window_samples, total_samples)
        chunk = audio_array[start_sample:end_sample]
        chunk_duration = len(chunk) / sampling_rate

        if chunk_duration < min_window_seconds and segments:
            continue

        result = pipeline.transcribe(
            chunk,
            sampling_rate=sampling_rate,
            **kwargs
        )
        segment_text = result.get("text", "").strip()
        total_inference_time += result.get("inference_time", 0.0)

        segments.append({
            "index": segment_index,
            "start_time": start_sample / sampling_rate,
            "end_time": end_sample / sampling_rate,
            "audio_duration": chunk_duration,
            "inference_time": result.get("inference_time", 0.0),
            "rtf": result.get("rtf", 0.0),
            "text": segment_text,
        })

        if segment_text:
            combined_text_parts.append(segment_text)

    combined_text = " ".join(combined_text_parts).strip()

    return {
        "text": combined_text,
        "audio_duration": audio_duration,
        "inference_time": total_inference_time,
        "rtf": total_inference_time / audio_duration if audio_duration > 0 else 0.0,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "segments": segments,
    }


class ManualONNXInference:
    """Manual ONNX Runtime inference for maximum speed."""

    def __init__(self, model_dir: str):
        """
        Initialize manual ONNX inference.

        Args:
            model_dir: Path to directory containing encoder_model.onnx and decoder_model.onnx
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for ONNX mode. Install with: uv pip install onnxruntime"
            )

        import os
        from transformers import AutoProcessor

        self.model_dir = Path(model_dir)

        # Find ONNX files
        encoder_path = self.model_dir / 'encoder_model.onnx'
        decoder_path = self.model_dir / 'decoder_model.onnx'

        # Check for alternate names
        if not encoder_path.exists():
            encoder_path = self.model_dir / 'encoder.onnx'
        if not decoder_path.exists():
            alt_decoder = self.model_dir / 'decoder.onnx'
            if alt_decoder.exists():
                decoder_path = alt_decoder
            else:
                decoder_path = self.model_dir / 'decoder_model_merged.onnx'

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder not found at {decoder_path}")

        print(f"Loading ONNX encoder: {encoder_path}")
        self.encoder_session = ort.InferenceSession(str(encoder_path))

        print(f"Loading ONNX decoder: {decoder_path}")
        self.decoder_session = ort.InferenceSession(str(decoder_path))

        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        except Exception:
            print("Loading processor from base model")
            self.processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")

        # Token IDs
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 2

        print("✅ Manual ONNX model loaded")

    def encode(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """Encode audio to hidden states."""
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np"
        )
        input_values = inputs.input_values

        # Get encoder input names
        encoder_inputs = {inp.name: None for inp in self.encoder_session.get_inputs()}

        if 'input_values' in encoder_inputs:
            encoder_inputs['input_values'] = input_values
        elif 'input_features' in encoder_inputs:
            encoder_inputs['input_features'] = input_values

        # Run encoder
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        return encoder_outputs[0]

    def decode_greedy(self, encoder_hidden_states: np.ndarray, max_new_tokens: int = 50) -> np.ndarray:
        """Greedy decoding from encoder hidden states."""
        decoder_input_ids = np.array([[self.bos_token_id]], dtype=np.int64)
        generated_tokens = []

        # Get decoder input names
        decoder_input_names = [inp.name for inp in self.decoder_session.get_inputs()]

        for _ in range(max_new_tokens):
            # Prepare decoder inputs
            decoder_inputs = {}
            for name in decoder_input_names:
                if 'input_ids' in name:
                    decoder_inputs[name] = decoder_input_ids
                elif 'encoder_hidden_states' in name or 'encoder_outputs' in name:
                    decoder_inputs[name] = encoder_hidden_states
                elif 'encoder_attention_mask' in name:
                    decoder_inputs[name] = np.ones(
                        (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]),
                        dtype=np.int64
                    )

            # Run decoder
            logits = self.decoder_session.run(None, decoder_inputs)[0]

            # Get next token (greedy)
            next_token = np.argmax(logits[:, -1, :], axis=-1)

            # Check for EOS
            if next_token[0] == self.eos_token_id:
                break

            generated_tokens.append(next_token[0])

            # Update decoder input
            decoder_input_ids = np.concatenate([
                decoder_input_ids,
                next_token.reshape(1, 1)
            ], axis=1)

        return np.array([generated_tokens])

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        sampling_rate: int = 16000,
        max_new_tokens: Optional[int] = None,
        **kwargs  # Accept but ignore PyTorch-specific kwargs
    ) -> Dict:
        """
        Transcribe audio using ONNX Runtime.

        Args:
            audio: Audio array or path to audio file
            sampling_rate: Audio sampling rate
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'text', 'audio_duration', 'inference_time', 'rtf'
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            audio_array = load_audio(audio_path, target_sr=sampling_rate)
            audio_duration = len(audio_array) / sampling_rate
        else:
            audio_array = audio
            audio_duration = len(audio_array) / sampling_rate

        # Normalize audio
        audio_array = normalize_audio(audio_array, target_rms=0.075)

        # Calculate max_new_tokens if not provided
        if max_new_tokens is None:
            max_new_tokens = max(10, min(int(audio_duration * 5), 150))

        # Transcribe
        start_time = time.time()

        encoder_hidden_states = self.encode(audio_array, sampling_rate)
        token_ids = self.decode_greedy(encoder_hidden_states, max_new_tokens)
        transcription = self.processor.tokenizer.decode(
            token_ids[0],
            skip_special_tokens=True
        )

        inference_time = time.time() - start_time

        return {
            'text': transcription.strip(),
            'audio_duration': audio_duration,
            'inference_time': inference_time,
            'rtf': inference_time / audio_duration
        }

    def transcribe_batch(
        self,
        audio_paths: List[Path],
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict]:
        """Transcribe multiple audio files."""
        results = []
        iterator = tqdm(audio_paths, desc="Transcribing (ONNX)") if show_progress else audio_paths

        for audio_path in iterator:
            try:
                result = self.transcribe(audio_path, **kwargs)
                result['file'] = str(audio_path)
                results.append(result)
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")
                results.append({
                    'file': str(audio_path),
                    'text': '',
                    'error': str(e)
                })

        return results


class MoonshineInference:
    """Moonshine ASR inference wrapper."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        fp16: bool = False
    ):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to model directory or HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            fp16: Use FP16 precision (only on CUDA)
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.fp16 = fp16 and self.device.type == "cuda"

        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        print(f"FP16: {self.fp16}")

        # Load model and processor
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Move to device
        self.model.to(self.device)

        # Enable FP16 if requested
        if self.fp16:
            self.model.half()

        self.model.eval()

        print(f"Model loaded: {self.model.num_parameters():,} parameters")
        print(f"Vocab size: {self.processor.tokenizer.vocab_size}")

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        sampling_rate: int = 16000,
        num_beams: int = 5,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 2,
        return_timestamps: bool = False,
        max_new_tokens: Optional[int] = None
    ) -> Dict:
        """
        Transcribe audio file or array.

        Args:
            audio: Audio array or path to audio file
            sampling_rate: Audio sampling rate (if array provided)
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeating tokens
            no_repeat_ngram_size: Block repeated n-grams
            return_timestamps: Return word-level timestamps (not supported yet)
            max_new_tokens: Maximum tokens to generate (auto if None)

        Returns:
            Dictionary with 'text', 'duration', 'inference_time'
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            audio_array = load_audio(audio_path, target_sr=sampling_rate)
            audio_duration = len(audio_array) / sampling_rate
        else:
            audio_array = audio
            audio_duration = len(audio_array) / sampling_rate

        # Normalize audio
        audio_array = normalize_audio(audio_array, target_rms=0.075)

        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Convert to FP16 if enabled
        if self.fp16:
            input_values = input_values.half()

        # Calculate max_new_tokens based on audio duration if not provided
        if max_new_tokens is None:
            # Roughly 5 tokens per second (accounting for French verbosity)
            # Min 10 tokens, max 150 tokens
            max_new_tokens = max(10, min(int(audio_duration * 5), 150))

        # Transcribe
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                input_values=input_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=False,
                early_stopping=True
            )

        inference_time = time.time() - start_time

        # Decode
        transcription = self.processor.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        result = {
            'text': transcription.strip(),
            'audio_duration': audio_duration,
            'inference_time': inference_time,
            'rtf': inference_time / audio_duration  # Real-time factor
        }

        return result

    def transcribe_batch(
        self,
        audio_paths: List[Path],
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing (currently processes one at a time)
            show_progress: Show progress bar
            **kwargs: Additional arguments for transcribe()

        Returns:
            List of transcription results
        """
        results = []

        iterator = tqdm(audio_paths, desc="Transcribing") if show_progress else audio_paths

        for audio_path in iterator:
            try:
                result = self.transcribe(audio_path, **kwargs)
                result['file'] = str(audio_path)
                results.append(result)
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")
                results.append({
                    'file': str(audio_path),
                    'text': '',
                    'error': str(e)
                })

        return results


class LiveTranscriber:
    """Live audio transcription with optional VAD."""

    def __init__(
        self,
        inference_pipeline: MoonshineInference,
        use_vad: bool = True,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize live transcription.

        Args:
            inference_pipeline: MoonshineInference instance
            use_vad: Use Voice Activity Detection (Silero VAD)
            sample_rate: Audio sampling rate
            chunk_duration: Transcribe every N seconds (continuous mode)
            confidence_threshold: Minimum confidence for VAD
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is required for live mode. Install with: uv pip install sounddevice"
            )

        self.pipeline = inference_pipeline
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.confidence_threshold = confidence_threshold
        self.use_vad = use_vad

        # Buffers
        self.vad_buffer = np.array([])
        self.audio_buffer = np.array([])
        self.is_speaking = False

        # Load VAD if enabled
        self.vad_model = None
        self.vad_iterator = None

        if use_vad:
            print("Loading Silero VAD model...")
            try:
                self.vad_model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    source='github',
                    onnx=True
                )
                (get_speech_timestamps, save_audio, read_audio,
                 VADIterator, collect_chunks) = utils
                self.vad_iterator = VADIterator(self.vad_model)
                print("VAD model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load VAD model: {e}")
                print("Falling back to continuous mode")
                self.use_vad = False

    def _callback_with_vad(self, indata, frames, time_info, status):
        """Audio callback with Voice Activity Detection."""
        if status and str(status) != 'input overflow':
            print(f"Stream status: {status}")

        audio_flatten = indata[:, 0].flatten()

        # Accumulate audio if speaking
        if self.is_speaking:
            self.audio_buffer = np.append(self.audio_buffer, audio_flatten)

        # VAD buffer (process in chunks of ~3 * 1536 samples)
        self.vad_buffer = np.append(self.vad_buffer, audio_flatten)

        if len(self.vad_buffer) >= 1536 * 3:
            speech_dict = self.vad_iterator(self.vad_buffer, return_seconds=True)

            if speech_dict:
                if 'start' in speech_dict:
                    # Speech started
                    self.audio_buffer = np.copy(self.vad_buffer)
                    self.is_speaking = True
                    print("🎤 Speech detected...")

                elif 'end' in speech_dict:
                    # Speech ended - transcribe
                    self.is_speaking = False

                    if len(self.audio_buffer) > 0:
                        try:
                            start_time = time.time()
                            result = self.pipeline.transcribe(
                                self.audio_buffer,
                                sampling_rate=self.sample_rate
                            )
                            elapsed = time.time() - start_time

                            if result['text']:
                                print(f"📝 {result['text']}")
                                print(f"   (RTF: {result['rtf']:.2f}x, {elapsed:.2f}s)")

                        except Exception as e:
                            print(f"Transcription error: {e}")

                        # Clear buffer
                        self.audio_buffer = np.array([])

            # Clear VAD buffer after processing
            self.vad_buffer = np.array([])

    def _callback_continuous(self, indata, frames, time_info, status):
        """Audio callback for continuous transcription (no VAD)."""
        if status and str(status) != 'input overflow':
            print(f"Stream status: {status}")

        audio_flatten = indata[:, 0].flatten()
        self.audio_buffer = np.append(self.audio_buffer, audio_flatten)

        # Transcribe every chunk_duration seconds
        if len(self.audio_buffer) >= self.sample_rate * self.chunk_duration:
            try:
                start_time = time.time()
                result = self.pipeline.transcribe(
                    self.audio_buffer,
                    sampling_rate=self.sample_rate
                )
                elapsed = time.time() - start_time

                if result['text']:
                    print(f"📝 {result['text']}")
                    print(f"   (RTF: {result['rtf']:.2f}x, {elapsed:.2f}s)")

            except Exception as e:
                print(f"Transcription error: {e}")

            # Clear buffer
            self.audio_buffer = np.array([])

    def start(self):
        """Start live transcription."""
        callback = self._callback_with_vad if self.use_vad else self._callback_continuous

        print("\n" + "=" * 60)
        print("LIVE TRANSCRIPTION MODE")
        print("=" * 60)
        print(f"VAD: {'Enabled' if self.use_vad else 'Disabled'}")
        print(f"Sample rate: {self.sample_rate} Hz")
        if not self.use_vad:
            print(f"Chunk duration: {self.chunk_duration}s")
        print("=" * 60)
        print("🎙️  Listening... Press Ctrl+C to stop.\n")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=callback
            ):
                while True:
                    sd.sleep(1000)

        except KeyboardInterrupt:
            print("\n\n✅ Stopped by user")
        except Exception as e:
            print(f"\n❌ Error with input stream: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Moonshine ASR Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python inference.py --model ./results-moonshine-fr/final --audio sample.wav

  # Directory of files
  python inference.py --model ./model --audio ./test_audio/ --output results.json

  # Custom generation parameters
  python inference.py --model ./model --audio audio.wav \\
      --num-beams 5 --repetition-penalty 1.3 --no-repeat-ngram-size 2

  # GPU with FP16
  python inference.py --model ./model --audio audio.wav --device cuda --fp16

  # Live transcription with VAD (recommended)
  python inference.py --model ./model --live

  # Live transcription without VAD (continuous mode)
  python inference.py --model ./model --live --no-vad --chunk-duration 2.0

  # ONNX Runtime for super fast inference (requires ONNX model)
  python inference.py --model ./model-onnx --audio audio.wav --onnx

  # Manual ONNX (fastest)
  python inference.py --model ./model-onnx --audio audio.wav --use-manual-onnx
        """
    )

    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model directory or HuggingFace model name'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file or directory containing audio files (not required for --live mode)'
    )

    # Optional arguments
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results (default: print to stdout)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use FP16 precision (CUDA only, faster inference)'
    )

    # Generation parameters
    parser.add_argument(
        '--num-beams',
        type=int,
        default=5,
        help='Number of beams for beam search (default: 5)'
    )
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.3,
        help='Penalty for repeating tokens (default: 1.3)'
    )
    parser.add_argument(
        '--no-repeat-ngram-size',
        type=int,
        default=2,
        help='Block repeated n-grams (default: 2)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        help='Maximum tokens to generate (default: auto based on duration)'
    )
    parser.add_argument(
        '--window-seconds',
        type=float,
        help='Optional fixed window size for simple sliding-window transcription'
    )
    parser.add_argument(
        '--window-stride-seconds',
        type=float,
        help='Optional stride for sliding-window transcription (default: same as window size)'
    )

    # Live mode arguments
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live transcription from microphone'
    )
    parser.add_argument(
        '--no-vad',
        action='store_true',
        help='Disable Voice Activity Detection in live mode (continuous transcription)'
    )
    parser.add_argument(
        '--chunk-duration',
        type=float,
        default=2.0,
        help='Chunk duration in seconds for continuous mode (default: 2.0)'
    )

    # ONNX mode arguments
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Use ONNX Runtime for super fast inference (requires ONNX model)'
    )
    parser.add_argument(
        '--use-manual-onnx',
        action='store_true',
        help='Use manual ONNX inference (fastest, lower-level)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.live and not args.audio:
        parser.error("Either --audio or --live must be specified")

    if args.live and args.audio:
        print("Warning: --audio is ignored in live mode")

    # Initialize inference pipeline
    if args.onnx or args.use_manual_onnx:
        # ONNX Runtime mode (super fast!)
        print("\n🚀 Using ONNX Runtime for maximum speed")

        if args.use_manual_onnx or not OPTIMUM_AVAILABLE:
            # Manual ONNX inference (fastest)
            if args.use_manual_onnx and not OPTIMUM_AVAILABLE:
                print("Note: Optimum not available, using manual ONNX")
            pipeline = ManualONNXInference(model_dir=args.model)
        else:
            # Optimum-based ONNX inference (easier)
            print("Using Optimum ORTModel")
            try:
                pipeline = ORTModelForSpeechSeq2Seq.from_pretrained(args.model)
                # Create wrapper to match interface
                class OptimumWrapper:
                    def __init__(self, model, model_path):
                        self.model = model
                        from transformers import AutoProcessor
                        try:
                            self.processor = AutoProcessor.from_pretrained(model_path)
                        except:
                            self.processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")

                    def transcribe(self, audio, sampling_rate=16000, max_new_tokens=None, **kwargs):
                        """Transcribe using Optimum."""
                        if isinstance(audio, (str, Path)):
                            audio_path = Path(audio)
                            audio_array = load_audio(audio_path, target_sr=sampling_rate)
                            audio_duration = len(audio_array) / sampling_rate
                        else:
                            audio_array = audio
                            audio_duration = len(audio_array) / sampling_rate

                        # Normalize
                        audio_array = normalize_audio(audio_array, target_rms=0.075)

                        # Process
                        inputs = self.processor(
                            audio_array,
                            sampling_rate=sampling_rate,
                            return_tensors="pt",
                            return_attention_mask=True
                        )

                        if max_new_tokens is None:
                            max_new_tokens = max(10, min(int(audio_duration * 5), 150))

                        # Transcribe
                        start_time = time.time()
                        generated_ids = self.model.generate(
                            inputs.input_values,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_new_tokens
                        )
                        inference_time = time.time() - start_time

                        transcription = self.processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0]

                        return {
                            'text': transcription.strip(),
                            'audio_duration': audio_duration,
                            'inference_time': inference_time,
                            'rtf': inference_time / audio_duration
                        }

                    def transcribe_batch(self, audio_paths, show_progress=True, **kwargs):
                        """Batch transcription."""
                        results = []
                        iterator = tqdm(audio_paths, desc="Transcribing (Optimum ONNX)") if show_progress else audio_paths
                        for audio_path in iterator:
                            try:
                                result = self.transcribe(audio_path, **kwargs)
                                result['file'] = str(audio_path)
                                results.append(result)
                            except Exception as e:
                                print(f"\nError: {e}")
                                results.append({'file': str(audio_path), 'text': '', 'error': str(e)})
                        return results

                pipeline = OptimumWrapper(pipeline, args.model)

            except Exception as e:
                print(f"Optimum loading failed: {e}")
                print("Falling back to manual ONNX inference")
                pipeline = ManualONNXInference(model_dir=args.model)
    else:
        # Standard PyTorch mode
        pipeline = MoonshineInference(
            model_path=args.model,
            device=args.device,
            fp16=args.fp16
        )

    # Live mode
    if args.live:
        live_transcriber = LiveTranscriber(
            inference_pipeline=pipeline,
            use_vad=not args.no_vad,
            chunk_duration=args.chunk_duration
        )
        live_transcriber.start()
        return 0

    # Process audio
    audio_path = Path(args.audio)

    if audio_path.is_file():
        # Single file
        print(f"\nTranscribing: {audio_path}")
        if args.window_seconds:
            result = transcribe_with_sliding_windows(
                pipeline,
                audio_path,
                window_seconds=args.window_seconds,
                stride_seconds=args.window_stride_seconds,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_new_tokens=args.max_new_tokens
            )
        else:
            result = pipeline.transcribe(
                audio_path,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_new_tokens=args.max_new_tokens
            )

        print(f"\nTranscription: {result['text']}")
        print(f"Audio duration: {result['audio_duration']:.2f}s")
        print(f"Inference time: {result['inference_time']:.2f}s")
        print(f"Real-time factor: {result['rtf']:.2f}x")
        if 'segments' in result:
            print("\nWindowed segments:")
            for segment in result['segments']:
                print(
                    f"[{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] "
                    f"{segment['text']}"
                )

        results = [result]
        results[0]['file'] = str(audio_path)

    elif audio_path.is_dir():
        # Directory of files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
            audio_files.extend(audio_path.glob(ext))
            audio_files.extend(audio_path.glob(ext.upper()))

        audio_files = sorted(set(audio_files))

        if not audio_files:
            print(f"No audio files found in {audio_path}")
            return 1

        print(f"\nFound {len(audio_files)} audio files")

        if args.window_seconds:
            results = []
            iterator = tqdm(audio_files, desc="Transcribing")
            for file_path in iterator:
                try:
                    result = transcribe_with_sliding_windows(
                        pipeline,
                        file_path,
                        window_seconds=args.window_seconds,
                        stride_seconds=args.window_stride_seconds,
                        num_beams=args.num_beams,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        max_new_tokens=args.max_new_tokens
                    )
                    result['file'] = str(file_path)
                    results.append(result)
                except Exception as e:
                    print(f"\nError processing {file_path}: {e}")
                    results.append({
                        'file': str(file_path),
                        'text': '',
                        'error': str(e)
                    })
        else:
            results = pipeline.transcribe_batch(
                audio_files,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_new_tokens=args.max_new_tokens
            )

        # Print summary
        print(f"\n{'='*60}")
        print(f"TRANSCRIPTION SUMMARY")
        print(f"{'='*60}")
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        print(f"Total files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if successful > 0:
            avg_rtf = np.mean([r['rtf'] for r in results if 'error' not in r])
            print(f"Average real-time factor: {avg_rtf:.2f}x")

    else:
        print(f"Error: {audio_path} is neither a file nor a directory")
        return 1

    # Save or print results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")
    else:
        print(f"\n{'='*60}")
        print("RESULTS (JSON)")
        print(f"{'='*60}")
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
