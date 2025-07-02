import os
from pathlib import Path
import tempfile
from datetime import datetime
from typing import Any

import gradio as gr

MODEL_CACHE_DIR = Path("./models")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import queue
import torch
import torch.multiprocessing as mp
from multiprocessing.pool import Pool
from multiprocessing.queues import Queue
from dataclasses import dataclass, asdict, replace, field, fields
import shutil
import random
import time
import gc

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mp.set_start_method("spawn", force=True)

TEMP_PATH = Path("temp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(TEMP_PATH)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

LORA_PATH = Path("loras")
LORA_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUT_PATH = OUTPUT_PATH / "video"
VIDEO_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

INPUT_PATH = Path("inputs")
INPUT_PATH.mkdir(parents=True, exist_ok=True)
UPLOAD_INPUT_PATH = INPUT_PATH / "uploads"
UPLOAD_INPUT_PATH.mkdir(parents=True, exist_ok=True)
DATASET_INPUT_PATH = INPUT_PATH / "datasets"
DATASET_INPUT_PATH.mkdir(parents=True, exist_ok=True)

from echo_studio.utils import clean_memory_windows

def _init_worker(q_loacal: Queue):
    from safetensors.torch import save_file, load_file
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from transformers import (
        LlamaTokenizerFast,
        LlamaConfig,
        LlamaModel,
    )
    
    from musubi_tuner.fpack_generate_video import (
        load_text_encoder1,
        load_text_encoder2,
        prepare_image_inputs,
        prepare_text_inputs,
        merge_lora_weights,
        convert_lora_for_framepack,
        preprocess_magcache,
        sample_hunyuan,
        postprocess_magcache,
    )
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
    from musubi_tuner.frame_pack.hunyuan_video_packed_inference import HunyuanVideoTransformer3DModelPackedInference
    from musubi_tuner.frame_pack.framepack_utils import LLAMA_CONFIG
    from musubi_tuner.utils.safetensors_utils import load_safetensors
    from musubi_tuner.networks import lora_framepack
    global q
    q = q_loacal

@dataclass
class Dataset:
    video_directory: str = None
    video_jsonl_file: str = None
    image_directory: str = None
    image_jsonl_file: str = None
    control_directory: str = None
    cache_directory: str = None
    num_repeats: int = 1
    frame_extraction: str = "full"
    # head: 動画から最初のNフレームを抽出します。
    # chunk: 動画をNフレームずつに分割してフレームを抽出します。
    # slide: frame_strideに指定したフレームごとに動画からNフレームを抽出します。
    # uniform: 動画から一定間隔で、frame_sample個のNフレームを抽出します。
    # full: 動画から全てのフレームを抽出します。
    frame_stride = None
    frame_sample = None
    target_frames: list[int] = None
    max_frames = None
    source_fps: float = None
    batch_size: int = None
    num_repeats: int = None
    enable_bucket: bool = None
    bucket_no_upscale: bool = None

    # FramePack
    fp_latent_window_size: int = 9
    # one frame inference
    fp_1f_clean_indices: list[int] = field(default_factory=lambda: [0])
    fp_1f_target_index: int = 9
    no_post: bool = False
    fp_1f_clean_indices: list[int] = None
    fp_1f_target_index: int = None
    fp_1f_no_post: bool = False

@dataclass
class GeneralConfig:
    # FramePack default 640x640
    resolution: list[int] = field(default_factory=lambda: [480, 832])
    caption_extension:str = ".txt"
    batch_size: int = 1
    num_repeats: int = 1
    enable_bucket: bool = True
    bucket_no_upscale: bool = False

@dataclass
class DatasetConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    datasets: list[Dataset] = field(default_factory=lambda: [])

@dataclass
class Args:
    dit: str = None
    dit_f1: str = None
    vae: str = None
    text_encoder1: str = None
    text_encoder2: str = None
    image_encoder: str = None
    f1: bool = False
    device: torch.device = device

    lora_weight: str = None
    lora_multiplier: float = 1.0
    include_patterns: str = None
    exclude_patterns: str = None
    lycoris: bool = False
    save_merged_model: bool = False

    one_frame_inference: str = None
    sample_solver: str = "unipc"
    video_seconds: float = None
    video_sections: int = 1
    video_size: list[int] = field(default_factory=lambda: [480, 832])
    fps: int = 24
    infer_steps: int = 25
    save_path: str = None
    output_type: str = "video"
    no_metadata: bool = False
    seed: int = None
    attn_mode: str = "sageattn"
    blocks_to_swap: int = 24
    vae_chunk_size: int = 16
    vae_spatial_tile_sample_min_size: int = 128

    from_file: bool = False
    interactive: bool = False

    prompt: str = None
    negative_prompt: str = None
    custom_system_prompt: str = None

    image_path: str = None
    latent_path: str = None
    control_image_path: str = None
    control_image_mask_path: str = None
    end_image_path: str = None

    magcache_mag_ratios: str = None
    magcache_retention_ratio: float = 0.2
    magcache_threshold: float = 0.24
    magcache_k: int = 6
    magcache_calibration: bool = False

    latent_window_size: int = 9
    embedded_cfg_scale: float = 10.0
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0

    flow_shift: float = None
    latent_paddings: str = None

    fp8_scaled: bool = True
    fp8: bool = False
    fp8_llm: bool = True

    rope_scaling_factor: float = 0.5
    rope_scaling_timestep_threshold: int = None

    bulk_decode: bool = False

def get_save_path() -> Path:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    save_path = VIDEO_OUTPUT_PATH / date_str
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path

class FramePackUI:
    NORMAL_INFERENCE = "通常推論"
    ONE_FRAME_INFERENCE = "1フレーム推論"
    SECTION_NUMBER_MAX = 10
    def __init__(self, framepack):
        self.framepack: FramePack = framepack

        self.inference_mode = gr.State(FramePackUI.NORMAL_INFERENCE)

        self.mode_tabs = gr.Tabs()

        self.message = gr.Textbox(label="動画生成ログ")
        self.inference_mode_tabs = gr.Tabs()
        self.video_tab = gr.Tab(FramePackUI.NORMAL_INFERENCE)
        self.one_frame_inference_tab = gr.Tab(FramePackUI.ONE_FRAME_INFERENCE)
        self.f1_mode_enable = gr.Checkbox(False, label="F1推論")
        self.start_image_path = gr.Image(label="開始画像", type="filepath")
        self.start_image_path_state = gr.State(None)
        self.latent_path = gr.File(label="latent", type="filepath")

        self.advanced_video_control_panel = gr.Accordion("高度な動画制御", open=False)
        self.end_image_path = gr.Image(label="終端画像", type="filepath")

        self.section_areas: list[gr.Group] = []
        self.section_titles: list[gr.Markdown] = []
        self.section_start_image_paths: list[gr.Image] = []
        self.section_start_image_path_states: list[gr.State] = []
        self.section_prompt_texts: list[gr.TextArea] = []
        self.section_padding_number: list[gr.Number] = []
        init_sections = self.framepack.args.video_sections if self.framepack.args else 1
        for n in range(FramePackUI.SECTION_NUMBER_MAX):
            isVisible = init_sections > n
            self.section_areas.append(gr.Group(visible=isVisible))
            self.section_titles.append(gr.Markdown(f"### セクション{n+1}", padding=True))
            self.section_start_image_paths.append(gr.Image(label="画像ガイダンス", type="filepath", interactive=True))
            self.section_start_image_path_states.append(gr.State(None))
            self.section_prompt_texts.append(gr.TextArea(
                "",
                label="プロンプト",
                interactive=True,
            ))
            self.section_padding_number.append(gr.Number(1, label="padding", visible=False))
        self.section_padding_enable = gr.Checkbox(False, label="セクション padding 設定")

        self.prompt_text = gr.TextArea(
            "The camera rotates to face the center of the scene while remaining at a constant distance. Nothing moves except the camera.",
            label="プロンプト",
        )
        # self.negative_prompt_text = gr.TextArea(label="ネガティブプロンプト")
        self.blocks_to_swap_number = gr.Number(
            value=self.framepack.args.blocks_to_swap if self.framepack.args else 24,
            label="BlocksToSwap", minimum=0, maximum=38
        )
        self.video_seconds_number = gr.Number(
            value=self.framepack.args.video_seconds if self.framepack.args else None,
            label="VideoSeconds", minimum=0, maximum=10
        )
        self.video_sections_number = gr.Number(
            value=self.framepack.args.video_sections if self.framepack.args else 1,
            label="VideoSections", minimum=0, maximum=FramePackUI.SECTION_NUMBER_MAX
        )
        self.vae_chunk_size_number = gr.Number(
            value=self.framepack.args.vae_chunk_size if self.framepack.args else 16,
            label="VaeChunkSize"
        )
        self.vae_spatial_tile_sample_min_size_number = gr.Number(
            value=self.framepack.args.vae_spatial_tile_sample_min_size if self.framepack.args else 128,
            label="VaeSpatialTileSampleMinSize",
        )
        self.size_width_number = gr.Number(
            value=self.framepack.args.video_size[1] if self.framepack.args else 768,
            label="VideoSizeWidth",
        )
        self.size_height_number = gr.Number(
            value=self.framepack.args.video_size[0] if self.framepack.args else 512,
            label="VideoSizeHeight",
        )
        self.size_swap_btn = gr.Button("🔁")
        self.one_frame_inference_text = gr.Textbox(
            value="default",
            label="1フレーム推論 オプション", visible=False,
        )
        self.seed_number = gr.Number(
            value=self.framepack.args.seed if self.framepack.args else 0,
            label="seed", minimum=-1,
        )
        self.output_image_gallery = gr.Gallery(label="生成画像", format="png", object_fit="contain", columns=1)
        self.output_video_path = gr.Video(label="生成動画")
        self.generate_btn = gr.Button("生成")
        self.decode_btn = gr.Button("デコード")

    def reg_events(self, demo: gr.Blocks):
        print("reg_events", self.framepack)
        with demo:
            def on_select_mode_change(evt: gr.SelectData):
                if evt.selected:
                    return evt.value
            def on_toggle_one_frame_inference_option(inference_mode):
                return gr.update(visible=(inference_mode == FramePackUI.ONE_FRAME_INFERENCE))
            self.video_tab.select(
                on_select_mode_change, inputs=None, outputs=[self.inference_mode]
            ).then(on_toggle_one_frame_inference_option, inputs=[self.inference_mode], outputs=[self.one_frame_inference_text])
            self.one_frame_inference_tab.select(
                on_select_mode_change, inputs=None, outputs=[self.inference_mode]
            ).then(on_toggle_one_frame_inference_option, inputs=[self.inference_mode], outputs=[self.one_frame_inference_text])

            def on_change_start_image_path(image_path):
                if image_path is not None:
                    image_path = Path(image_path)
                    upload_path = UPLOAD_INPUT_PATH / image_path.name
                    shutil.copy2(image_path, upload_path)
                    return str(upload_path)
                return None
            self.start_image_path.change(
                on_change_start_image_path,
                inputs=[self.start_image_path],
                outputs=[self.start_image_path_state],
            )
            for n in range(FramePackUI.SECTION_NUMBER_MAX):
                self.section_start_image_paths[n].change(
                    on_change_start_image_path,
                    inputs=[self.section_start_image_paths[n]],
                    outputs=[self.section_start_image_path_states[n]]
                )

            def size_swap(size_width, size_height):
                return gr.update(value=size_height), gr.update(value=size_width)
            self.size_swap_btn.click(
                size_swap,
                inputs=[self.size_width_number, self.size_height_number],
                outputs=[self.size_width_number, self.size_height_number]
            )

            def on_change_sections(video_sections):
                section_areas = []
                for n in range(FramePackUI.SECTION_NUMBER_MAX):
                    isVisible = video_sections > n
                    section_areas.append(gr.update(visible=isVisible))

                return [
                    *section_areas,
                ]
            self.video_sections_number.change(
                on_change_sections,
                inputs=[self.video_sections_number],
                outputs=[
                    *self.section_areas,
                ]
            )

            def on_change_section_padding_enable(section_padding_enable):
                section_padding_number = []
                for n in range(FramePackUI.SECTION_NUMBER_MAX):
                    section_padding_number.append(gr.update(visible=section_padding_enable))
                return [*section_padding_number]

            self.section_padding_enable.change(
                on_change_section_padding_enable,
                inputs=[self.section_padding_enable],
                outputs=[*self.section_padding_number],
            )

            def on_click_generate(
                f1_mode_enable,
                start_image,
                end_image,
                prompt,
                blocks_to_swap,
                video_seconds,
                video_sections,
                inference_mode,
                one_frame_inference,
                seed,
                size_width, size_height,
                *section_params,
            ):
                print("on_click_generate", self.framepack)
                if self.framepack is not None:
                    clean_memory_windows(target_ratio=0.5)
                    print("inference_mode", inference_mode)

                    section_start_image_path_states = list(section_params[:FramePackUI.SECTION_NUMBER_MAX])
                    section_prompt_texts = list(section_params[FramePackUI.SECTION_NUMBER_MAX:FramePackUI.SECTION_NUMBER_MAX * 2])

                    if self.framepack.args.video_seconds is None and self.framepack.args.video_sections is None:
                        # 元は5秒だけど、処理が長くなるので、デフォルトを1秒に
                        self.framepack.args.video_seconds = 1

                    self.framepack.args.image_path = start_image
                    self.framepack.args.prompt = prompt

                    if self.framepack.args.video_sections > 1:
                        image_paths = []
                        prompts = []
                        for n in range(self.framepack.args.video_sections):
                            if n == 0:
                                if section_start_image_path_states[n] is None or section_start_image_path_states[n] == "":
                                    section_start_image_path_states[n] = start_image
                                if section_prompt_texts[n] == "":
                                    section_prompt_texts[n] = prompt
                            if section_start_image_path_states[n] is not None and section_start_image_path_states[n] != "":
                                image_paths.append(f"{n}:{section_start_image_path_states[n]}")
                            if section_prompt_texts[n] != "":
                                prompts.append(f"{n}:{section_prompt_texts[n]}")
                        if len(image_paths) > 1:
                            self.framepack.args.image_path = ";;;".join(image_paths)
                        if len(prompts) > 1:
                            self.framepack.args.prompt = ";;;".join(prompts)

                    self.framepack.args.end_image_path = end_image

                    if inference_mode == FramePackUI.ONE_FRAME_INFERENCE:
                        self.framepack.args.one_frame_inference = one_frame_inference
                        self.framepack.args.control_image_path = [start_image]
                        self.framepack.args.end_image_path = None
                        self.framepack.args.output_type = "images"
                        self.framepack.args.save_path = str(get_save_path() / Path(start_image).stem)
                    else:
                        self.framepack.args.one_frame_inference = None
                        self.framepack.args.output_type = "video"

                    self.framepack.args.blocks_to_swap = blocks_to_swap
                    self.framepack.args.video_seconds = video_seconds if video_seconds > 0 else None
                    self.framepack.args.video_sections = video_sections if video_sections > 0 else None
                    self.framepack.args.seed = seed if seed >= 0 else None
                    self.framepack.args.video_size = [size_height, size_width]

                    self.framepack.args.lora_weight = [
                        str(LORA_PATH / "clockwise_V3_dim4_1e-3_640_640.safetensors"),
                    ]
                    self.framepack.args.lora_multiplier = [
                        1.0,
                    ]

                    ctx = mp.get_context("spawn")
                    q = ctx.Queue()
                    with ctx.Pool(processes=1, maxtasksperchild=5, initializer=_init_worker, initargs=(q,)) as pool:
                        print("models_download_worker")
                        for message in self.framepack.models_download_worker(pool, q):
                            yield gr.update(value=message), gr.update(), gr.update()
                        
                        if f1_mode_enable:
                            print("f1 mode...")
                            self.framepack.args.dit = self.framepack.args.dit_f1
                            self.framepack.args.f1 = f1_mode_enable
                        
                        print("convert_worker")
                        for message in self.framepack.convert_worker(pool, q):
                            yield gr.update(value=message), gr.update(), gr.update()
                        
                        print("prepare_i2v_inputs_worker")
                        for message in self.framepack.prepare_i2v_inputs_worker(pool, q):
                            yield gr.update(value=message), gr.update(), gr.update()
                        
                        if inference_mode == FramePackUI.ONE_FRAME_INFERENCE:
                            print("one_frame_inference")
                            for message, latent_path in self.framepack.one_frame_inference(pool, q):
                                if latent_path is None:
                                    yield gr.update(value=message), gr.update(), gr.update()
                                else:
                                    yield gr.update(), gr.update(), gr.update()
                        else:
                            print("generate_video_worker")
                            for message, latent_path in self.framepack.generate_video_worker(pool, q):
                                if latent_path is None:
                                    yield gr.update(value=message), gr.update(), gr.update()
                                else:
                                    yield gr.update(), gr.update(), gr.update()
                    
                        print("decode_latent_worker")
                        for message, save_path in self.framepack.decode_latent_worker(pool, q, latent_path):
                            if save_path is None:
                                yield gr.update(value=message), gr.update(), gr.update()
                            else:
                                if self.framepack.args.output_type == "video":
                                    yield gr.update(), gr.update(value=save_path), gr.update()
                                elif self.framepack.args.output_type == "images":
                                    yield gr.update(), gr.update(), gr.update(value=save_path, columns=len(save_path))

            self.generate_btn.click(
                on_click_generate,
                inputs=[
                    self.f1_mode_enable,
                    self.start_image_path_state,
                    self.end_image_path,
                    self.prompt_text,
                    self.blocks_to_swap_number,
                    self.video_seconds_number,
                    self.video_sections_number,
                    self.inference_mode,
                    self.one_frame_inference_text,
                    self.seed_number,
                    self.size_width_number,
                    self.size_height_number,
                    *self.section_start_image_path_states,
                    *self.section_prompt_texts,
                ],
                outputs=[self.message, self.output_video_path, self.output_image_gallery],
            )

            def on_click_decode(
                latent_path,
                inference_mode,
            ):
                print("on_click_decode", self.framepack)
                if self.framepack is not None:
                    clean_memory_windows(target_ratio=0.5)
                    print("inference_mode", inference_mode)

                    if inference_mode == FramePackUI.ONE_FRAME_INFERENCE:
                        self.framepack.args.output_type = "images"
                        self.framepack.args.save_path = str(get_save_path() / Path(latent_path).stem)
                    else:
                        self.framepack.args.output_type = "video"

                    ctx = mp.get_context("spawn")
                    q = ctx.Queue()
                    with ctx.Pool(processes=1, maxtasksperchild=2, initializer=_init_worker, initargs=(q,)) as pool:
                        print("models_download_worker")
                        for message in self.framepack.models_download_worker(pool, q):
                            yield gr.update(value=message), gr.update(), gr.update()
                        
                        print("decode_latent_worker")
                        for message, save_path in self.framepack.decode_latent_worker(pool, q, latent_path):
                            if save_path is None:
                                yield gr.update(value=message), gr.update(), gr.update()
                            else:
                                if self.framepack.args.output_type == "video":
                                    yield gr.update(), gr.update(value=save_path), gr.update()
                                elif self.framepack.args.output_type == "images":
                                    yield gr.update(), gr.update(), gr.update(value=save_path)

            self.decode_btn.click(
                on_click_decode,
                inputs=[
                    self.latent_path,
                    self.inference_mode,
                ],
                outputs=[self.message, self.output_video_path, self.output_image_gallery],
            )

def _models_download(args):
    global q
    log = f"[{os.getpid()}] worker started"
    print(log)
    q.put((False, log))

    from huggingface_hub import hf_hub_download

    log = f"repo_id: maybleMyers/framepack_h1111 -> FramePackI2V_HY_bf16.safetensors ..."
    print(log)
    q.put((False, log))
    args.dit = hf_hub_download(
        repo_id="maybleMyers/framepack_h1111",
        filename="FramePackI2V_HY_bf16.safetensors",
        cache_dir=MODEL_CACHE_DIR,
    )
    print(args.dit)
    q.put((False, args.dit))

    log = f"repo_id: maybleMyers/framepack_h1111 -> FramePack_F1_I2V_HY_20250503.safetensors ..."
    print(log)
    q.put((False, log))
    args.dit_f1 = hf_hub_download(
        repo_id="maybleMyers/framepack_h1111",
        filename="FramePack_F1_I2V_HY_20250503.safetensors",
        cache_dir=MODEL_CACHE_DIR,
    )
    print(args.dit_f1)
    q.put((False, args.dit_f1))

    log = f"repo_id: maybleMyers/framepack_h1111 -> pytorch_model.pt ..."
    print(log)
    q.put((False, log))
    args.vae = hf_hub_download(
        repo_id="maybleMyers/framepack_h1111",
        filename="pytorch_model.pt",
        cache_dir=MODEL_CACHE_DIR,
    )
    print(args.vae)
    q.put((False, args.vae))

    log = f"repo_id: maybleMyers/framepack_h1111 -> llava_llama3_fp16.safetensors ..."
    print(log)
    q.put((False, log))
    args.text_encoder1 = hf_hub_download(
        repo_id="maybleMyers/framepack_h1111",
        filename="llava_llama3_fp16.safetensors",
        cache_dir=MODEL_CACHE_DIR,
    )
    print(args.text_encoder1)
    q.put((False, args.text_encoder1))

    log = f"repo_id: maybleMyers/framepack_h1111 -> clip_l.safetensors ..."
    print(log)
    q.put((False, log))
    args.text_encoder2 = hf_hub_download(
        repo_id="maybleMyers/framepack_h1111",
        filename="clip_l.safetensors",
        cache_dir=MODEL_CACHE_DIR,
    )
    print(args.text_encoder2)
    q.put((False, args.text_encoder2))

    log = f"repo_id: maybleMyers/framepack_h1111 -> model.safetensors ..."
    print(log)
    q.put((False, log))
    args.image_encoder = hf_hub_download(
        repo_id="maybleMyers/framepack_h1111",
        filename="model.safetensors",
        cache_dir=MODEL_CACHE_DIR,
    )
    print(args.image_encoder)
    q.put((True, args.image_encoder))

def _convert_llm_fp8(fp_llm_fp8_path, args):
    global q, device
    log = f"[{os.getpid()}] worker started"
    print(log)
    q.put((False, log))

    from safetensors.torch import save_file
    from musubi_tuner.fpack_generate_video import load_text_encoder1
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device

    log = "Start convert LLM"
    print(log)
    q.put((False, log))

    if not fp_llm_fp8_path.exists():
        log = "Load LLM Model fp8"
        print(log)
        q.put((False, log))
        _, llm = load_text_encoder1(args, args.fp8_llm, device)
        state_dict = llm.state_dict()
        log = "Save state_dict LLM fp8"
        print(log)
        q.put((False, log))
        save_file(state_dict, fp_llm_fp8_path)
        del state_dict
        synchronize_device(device)
        clean_memory_on_device(device)
    
    log = "Complete convert LLM"
    print(log)
    q.put((True, log))

def _convert_dit_fp8_scaled(fp_dit_fp8_path, args):
    global q, device
    log = f"[{os.getpid()}] worker started"
    print(log)
    q.put((False, log))
    from safetensors.torch import save_file
    from accelerate import init_empty_weights
    from musubi_tuner.frame_pack.hunyuan_video_packed_inference import HunyuanVideoTransformer3DModelPackedInference
    from musubi_tuner.utils.safetensors_utils import load_safetensors
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device
    
    log = "Start convert DiT fp8_scaled"
    print(log)
    q.put((False, log))
    
    if not fp_dit_fp8_path.exists():
        with init_empty_weights():
            log = "Create DiT empty model"
            print(log)
            q.put((False, log))
            dit = HunyuanVideoTransformer3DModelPackedInference(
                attention_head_dim=128,
                guidance_embeds=True,
                has_clean_x_embedder=True,
                has_image_proj=True,
                image_proj_dim=1152,
                in_channels=16,
                mlp_ratio=4.0,
                num_attention_heads=24,
                num_layers=20,
                num_refiner_layers=2,
                num_single_layers=40,
                out_channels=16,
                patch_size=2,
                patch_size_t=1,
                pooled_projection_dim=768,
                qk_norm="rms_norm",
                rope_axes_dim=(16, 56, 56),
                rope_theta=256.0,
                text_embed_dim=4096,
                attn_mode=args.attn_mode,
                split_attn=False,
            )

        log = "Load state_dict"
        print(log)
        q.put((False, log))
        state_dict = load_safetensors(args.dit, device, disable_mmap=True, dtype=torch.bfloat16)

        log = "Optimization fp8 scaled"
        print(log)
        q.put((False, log))
        state_dict = dit.fp8_optimization(state_dict, device, move_to_device=False)

        log = "Save state_dict DiT fp8_scaled"
        print(log)
        q.put((False, log))
        save_file(state_dict, fp_dit_fp8_path)
        del state_dict
        synchronize_device(device)
        clean_memory_on_device(device)
    
    log = "Compete convert DiT fp8_scaled"
    print(log)
    q.put((True, log))

def _convert_dit_f1_fp8_scaled(fp_dit_f1_fp8_path, args):
    global q, device
    log = f"[{os.getpid()}] worker started"
    print(log)
    q.put((False, log))
    from safetensors.torch import save_file
    from accelerate import init_empty_weights
    from musubi_tuner.frame_pack.hunyuan_video_packed_inference import HunyuanVideoTransformer3DModelPackedInference
    from musubi_tuner.utils.safetensors_utils import load_safetensors
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device
    
    log = f"Start convert F1 DiT fp8_scaled {args.dit_f1}"
    print(log)
    q.put((False, log))
    
    if not fp_dit_f1_fp8_path.exists():
        with init_empty_weights():
            log = "Create F1 DiT empty model"
            print(log)
            q.put((False, log))
            dit = HunyuanVideoTransformer3DModelPackedInference(
                attention_head_dim=128,
                guidance_embeds=True,
                has_clean_x_embedder=True,
                has_image_proj=True,
                image_proj_dim=1152,
                in_channels=16,
                mlp_ratio=4.0,
                num_attention_heads=24,
                num_layers=20,
                num_refiner_layers=2,
                num_single_layers=40,
                out_channels=16,
                patch_size=2,
                patch_size_t=1,
                pooled_projection_dim=768,
                qk_norm="rms_norm",
                rope_axes_dim=(16, 56, 56),
                rope_theta=256.0,
                text_embed_dim=4096,
                attn_mode=args.attn_mode,
                split_attn=False,
            )

        log = "Load state_dict"
        print(log)
        q.put((False, log))
        state_dict = load_safetensors(args.dit_f1, device, disable_mmap=True, dtype=torch.bfloat16)

        log = "Optimization fp8 scaled"
        print(log)
        q.put((False, log))
        state_dict = dit.fp8_optimization(state_dict, device, move_to_device=False)

        log = "Save state_dict F1 DiT fp8_scaled"
        print(log)
        q.put((False, log))
        save_file(state_dict, fp_dit_f1_fp8_path)
        del state_dict
        synchronize_device(device)
        clean_memory_on_device(device)
    
    log = "Compete convert F1 DiT fp8_scaled"
    print(log)
    q.put((True, log))

def _prepare_i2v_inputs(fp_llm_fp8_path, args, precomputed_image_data, precomputed_text_data):
    global q, device
    log = f"[{os.getpid()}] worker started"
    print(log)
    q.put((False, log))
    from safetensors.torch import load_file
    from accelerate import init_empty_weights
    from transformers import (
        LlamaTokenizerFast,
        LlamaConfig,
        LlamaModel,
    )
    from musubi_tuner.fpack_generate_video import (
        prepare_image_inputs,
        prepare_text_inputs,
        load_text_encoder2,
        load_vae,
    )
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device
    from musubi_tuner.frame_pack.framepack_utils import LLAMA_CONFIG

    if not fp_llm_fp8_path.exists():
        log = f"{fp_llm_fp8_path} not found -> convert"
        print(log)
        q.put((False, log))
        _convert_llm_fp8(fp_llm_fp8_path, args)
    log = "Load tokenizer1"
    print(log)
    q.put((False, log))
    tokenizer1 = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")

    log = "Load text_encoder1"
    print(log)
    q.put((False, log))
    config = LlamaConfig(**LLAMA_CONFIG)
    with init_empty_weights():
        llm = LlamaModel._from_config(config, torch_dtype=torch.float16)
    llm.to(dtype=torch.float8_e4m3fn)

    def prepare_fp8(llama_model: LlamaModel, target_dtype):
        def forward_hook(module):
            def forward(hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                return module.weight.to(input_dtype) * hidden_states.to(input_dtype)

            return forward

        for module in llama_model.modules():
            if module.__class__.__name__ in ["Embedding"]:
                # print("set", module.__class__.__name__, "to", target_dtype)
                module.to(target_dtype)
            if module.__class__.__name__ in ["LlamaRMSNorm"]:
                # print("set", module.__class__.__name__, "hooks")
                module.forward = forward_hook(module)
    
    prepare_fp8(llm, torch.float16)
    state_dict = load_file(fp_llm_fp8_path, str(device))
    llm.load_state_dict(state_dict, strict=True, assign=True)
    llm.eval()

    state_dict = None
    del state_dict

    synchronize_device(device)
    clean_memory_on_device(device)

    log = "Load text_encoder2 & tokenizer2"
    print(log)
    q.put((False, log))

    tokenizer2, clip = load_text_encoder2(args)
    shared_models = {
        "tokenizer1": tokenizer1,
        "text_encoder1": llm,
        "tokenizer2": tokenizer2,
        "text_encoder2": clip,
    }

    log = "Load VAE"
    print(log)
    q.put((False, log))
    vae = load_vae(
        args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device
    )

    log = "prepare_i2v_inputs"
    print(log)
    q.put((False, log))
    image_data = prepare_image_inputs(args, device, vae, shared_models)
    text_data = prepare_text_inputs(args, device, shared_models)

    args.video_seconds = image_data["video_seconds"]

    precomputed_image_data["height"] = image_data["height"]
    precomputed_image_data["width"] = image_data["width"]
    precomputed_image_data["video_seconds"] = image_data["video_seconds"]
    precomputed_image_data["context_img"] = image_data["context_img"]
    precomputed_image_data["end_latent"] = image_data["end_latent"]
    precomputed_image_data["control_mask_images"] = image_data["control_mask_images"]
    precomputed_image_data["control_latents"] = image_data["control_latents"]

    precomputed_text_data["context"] = text_data["context"]
    precomputed_text_data["context_null"] = text_data["context_null"]

    shared_models.clear()
    image_data.clear()
    text_data.clear()
    del shared_models, tokenizer1, llm, tokenizer2, clip, vae, image_data, text_data
    synchronize_device(device)
    clean_memory_on_device(device)

    log = "Complete prepare_i2v_inputs"
    print(log)
    q.put((True, log))

def _prepare_generate(fp_dit_fp8_path, args, precomputed_image_data, precomputed_text_data):
    global q, device
    log = f"[{os.getpid()}] worker started"
    print(log)
    q.put((False, log))

    from safetensors.torch import load_file
    from accelerate import init_empty_weights
    from musubi_tuner.fpack_generate_video import (
        merge_lora_weights,
        convert_lora_for_framepack,
    )
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
    from musubi_tuner.frame_pack.hunyuan_video_packed_inference import HunyuanVideoTransformer3DModelPackedInference
    from musubi_tuner.networks import lora_framepack

    log = "Create DiT Model"
    print(log)
    q.put((False, log))
    with init_empty_weights():
        dit = HunyuanVideoTransformer3DModelPackedInference(
            attention_head_dim=128,
            guidance_embeds=True,
            has_clean_x_embedder=True,
            has_image_proj=True,
            image_proj_dim=1152,
            in_channels=16,
            mlp_ratio=4.0,
            num_attention_heads=24,
            num_layers=20,
            num_refiner_layers=2,
            num_single_layers=40,
            out_channels=16,
            patch_size=2,
            patch_size_t=1,
            pooled_projection_dim=768,
            qk_norm="rms_norm",
            rope_axes_dim=(16, 56, 56),
            rope_theta=256.0,
            text_embed_dim=4096,
            attn_mode=args.attn_mode,
            split_attn=False,
        )
    if not fp_dit_fp8_path.exists():
        log = f"{fp_dit_fp8_path} not found -> convert"
        print(log)
        q.put((False, log))
        _convert_dit_fp8_scaled(fp_dit_fp8_path, args)
    
    clean_memory_windows(target_ratio=0.6)

    log = f"Load DiT state_dict {fp_dit_fp8_path}"
    print(log)
    q.put((False, log))

    state_dict = load_file(fp_dit_fp8_path, "cpu")
    apply_fp8_monkey_patch(dit, state_dict, use_scaled_mm=False)
    
    dit.load_state_dict(state_dict, strict=True, assign=True)
    state_dict = None
    del state_dict

    merge_lora_weights(
        lora_framepack, dit, args, device, 
        convert_lora_for_framepack
    )

    if args.rope_scaling_timestep_threshold is not None:
        dit.enable_rope_scaling(args.rope_scaling_timestep_threshold, args.rope_scaling_factor)

    log = "Generate video latents"
    print(log)
    q.put((False, log))

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed

    height = precomputed_image_data["height"]
    width = precomputed_image_data["width"]
    video_seconds = precomputed_image_data["video_seconds"]
    context_img = precomputed_image_data["context_img"]
    end_latent = precomputed_image_data["end_latent"]
    control_latents = precomputed_image_data["control_latents"]
    control_mask_images = precomputed_image_data["control_mask_images"]

    context = precomputed_text_data["context"]
    context_null = precomputed_text_data["context_null"]

    dit.switch_block_swap_for_inference()
    dit.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
    dit.move_to_device_except_swap_blocks(device)
    dit.prepare_block_swap_before_forward()

    synchronize_device(device)
    clean_memory_on_device(device)

    # sampling
    latent_window_size = args.latent_window_size  # default is 9
    # ex: (5s * 30fps) / (9 * 4) = 4.16 -> 4 sections, 60s -> 1800 / 36 = 50 sections
    total_latent_sections = (video_seconds * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)
    num_frames = latent_window_size * 4 - 3

    logger.info(
        f"Video size: {height}x{width}@{video_seconds} (HxW@seconds), fps: {args.fps}, num sections: {total_latent_sections}, "
        f"infer_steps: {args.infer_steps}, frames per generation: {num_frames}"
    )

    return (
        dit,
        end_latent,
        total_latent_sections,
        latent_window_size,
        context_img,
        context,
        context_null,
        num_frames,
        seed_g,
        control_latents,
        control_mask_images,
        height,
        width,
    )

def _one_frame_inference(fp_dit_fp8_path, args, precomputed_image_data, precomputed_text_data):
    (
        dit,
        end_latent,
        total_latent_sections,
        latent_window_size,
        context_img,
        context,
        context_null,
        num_frames,
        seed_g,
        control_latents,
        control_mask_images,
        height,
        width,
    ) = _prepare_generate(fp_dit_fp8_path, args, precomputed_image_data, precomputed_text_data)

    from musubi_tuner.fpack_generate_video import (
        generate_with_one_frame_inference,
    )
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device

    one_frame_inference = set()
    for mode in args.one_frame_inference.split(","):
        one_frame_inference.add(mode.strip())
    
    # print("total_latent_sections", total_latent_sections, "context_img:", len(context_img), "context:", len(context))
    # for idx, data in context_img.items():
    #     print(f"Section {idx}:")
    #     for key, value in data.items():
    #         if isinstance(value, torch.Tensor):
    #             print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    #         else:
    #             print(f"  {key}: {value}")

    # for i, latent in enumerate(control_latents):
    #     print(f"control_latents[{i}]: shape={latent.shape}, dtype={latent.dtype}")
    
    real_history_latents = generate_with_one_frame_inference(
        args,
        dit,
        context,
        context_null,
        context_img,
        control_latents,
        control_mask_images,
        latent_window_size,
        height,
        width,
        device,
        seed_g,
        one_frame_inference,
    )

    # print("real_history_latents", real_history_latents.shape)

    # Only clean up shared models if they were created within this function
    wait_for_clean_memory = False
    if "dit" in locals():  # if model was loaded locally
        del dit
        synchronize_device(device)
        wait_for_clean_memory = True

    # wait for 5 seconds until block swap is done
    if wait_for_clean_memory and args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    return _save_latent(args, real_history_latents)

def _generate_video(fp_dit_fp8_path, args, precomputed_image_data, precomputed_text_data):
    (
        dit,
        end_latent,
        total_latent_sections,
        latent_window_size,
        context_img,
        context,
        context_null,
        num_frames,
        seed_g,
        control_latents,
        control_mask_images,
        height,
        width,
    ) = _prepare_generate(fp_dit_fp8_path, args, precomputed_image_data, precomputed_text_data)

    from musubi_tuner.fpack_generate_video import (
        preprocess_magcache,
        sample_hunyuan,
        postprocess_magcache,
    )
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device

    # prepare history latents
    history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32)
    if end_latent is not None:
        logger.info(f"Use end image(s): {args.end_image_path}")
        history_latents[:, :, :1] = end_latent.to(history_latents)

    # prepare clean latents and indices
    # Inverted Anti-drifting
    total_generated_latent_frames = 0
    latent_paddings = reversed(range(total_latent_sections))

    if total_latent_sections > 4:
        # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
        # items looks better than expanding it when total_latent_sections > 4
        # One can try to remove below trick and just
        # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
        # 4 sections: 3, 2, 1, 0. 50 sections: 3, 2, 2, ... 2, 1, 0
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

    if args.latent_paddings is not None:
        # parse user defined latent paddings
        user_latent_paddings = [int(x) for x in args.latent_paddings.split(",")]
        if len(user_latent_paddings) < total_latent_sections:
            print(
                f"User defined latent paddings length {len(user_latent_paddings)} does not match total sections {total_latent_sections}."
            )
            print(f"Use default paddings instead for unspecified sections.")
            latent_paddings[: len(user_latent_paddings)] = user_latent_paddings
        elif len(user_latent_paddings) > total_latent_sections:
            print(
                f"User defined latent paddings length {len(user_latent_paddings)} is greater than total sections {total_latent_sections}."
            )
            print(f"Use only first {total_latent_sections} paddings instead.")
            latent_paddings = user_latent_paddings[:total_latent_sections]
        else:
            latent_paddings = user_latent_paddings

    latent_paddings = list(latent_paddings)  # make sure it's a list
    for loop_index in range(total_latent_sections):
        latent_padding = latent_paddings[loop_index]

        # Inverted Anti-drifting
        section_index_reverse = loop_index  # 0, 1, 2, 3
        section_index = total_latent_sections - 1 - section_index_reverse  # 3, 2, 1, 0
        section_index_from_last = -(section_index_reverse + 1)  # -1, -2, -3, -4

        is_last_section = section_index == 0
        is_first_section = section_index_reverse == 0
        latent_padding_size = latent_padding * latent_window_size

        logger.info(f"latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}")

        # select start latent
        if section_index_from_last in context_img:
            image_index = section_index_from_last
        elif section_index in context_img:
            image_index = section_index
        else:
            image_index = 0

        start_latent = context_img[image_index]["start_latent"]
        image_path = context_img[image_index]["image_path"]
        if image_index != 0:  # use section image other than section 0
            logger.info(
                f"Apply experimental section image, latent_padding_size = {latent_padding_size}, image_path = {image_path}"
            )

        # Inverted Anti-drifting
        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        (
            clean_latent_indices_pre,
            blank_indices,
            latent_indices,
            clean_latent_indices_post,
            clean_latent_2x_indices,
            clean_latent_4x_indices,
        ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, : 1 + 2 + 16, :, :].split(
            [1, 2, 16], dim=2
        )
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

        # prepare conditioning inputs
        if section_index_from_last in context:
            prompt_index = section_index_from_last
        elif section_index in context:
            prompt_index = section_index
        else:
            prompt_index = 0

        context_for_index = context[prompt_index]
        # if args.section_prompts is not None:
        logger.info(f"Section {section_index}: {context_for_index['prompt']}")

        llama_vec = context_for_index["llama_vec"].to(device, dtype=torch.bfloat16)
        llama_attention_mask = context_for_index["llama_attention_mask"].to(device)
        clip_l_pooler = context_for_index["clip_l_pooler"].to(device, dtype=torch.bfloat16)

        image_encoder_last_hidden_state = context_img[image_index]["image_encoder_last_hidden_state"].to(
            device, dtype=torch.bfloat16
        )

        llama_vec_n = context_null["llama_vec"].to(device, dtype=torch.bfloat16)
        llama_attention_mask_n = context_null["llama_attention_mask"].to(device)
        clip_l_pooler_n = context_null["clip_l_pooler"].to(device, dtype=torch.bfloat16)

        preprocess_magcache(args, dit)

        generated_latents = sample_hunyuan(
            transformer=dit,
            sampler=args.sample_solver,
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=args.guidance_scale,
            distilled_guidance_scale=args.embedded_cfg_scale,
            guidance_rescale=args.guidance_rescale,
            shift=args.flow_shift,
            num_inference_steps=args.infer_steps,
            generator=seed_g,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=device,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
        )
        postprocess_magcache(args, dit)

        # concatenate generated latents
        total_generated_latent_frames += int(generated_latents.shape[2])
        # Inverted Anti-drifting: prepend generated latents to history latents
        if is_last_section:
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            total_generated_latent_frames += 1

        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

        logger.info(f"Generated. Latent shape {real_history_latents.shape}")
    
    # Only clean up shared models if they were created within this function
    wait_for_clean_memory = False
    if "dit" in locals():  # if model was loaded locally
        del dit
        synchronize_device(device)
        wait_for_clean_memory = True

    # wait for 5 seconds until block swap is done
    if wait_for_clean_memory and args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    return _save_latent(args, real_history_latents)

def _generate_f1_video(fp_dit_fp8_path, args, precomputed_image_data, precomputed_text_data):
    (
        dit,
        end_latent,
        total_latent_sections,
        latent_window_size,
        context_img,
        context,
        context_null,
        num_frames,
        seed_g,
        control_latents,
        control_mask_images,
        height,
        width,
    ) = _prepare_generate(fp_dit_fp8_path, args, precomputed_image_data, precomputed_text_data)

    from musubi_tuner.fpack_generate_video import (
        preprocess_magcache,
        sample_hunyuan,
        postprocess_magcache,
    )
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device

    # prepare history latents
    history_latents = torch.zeros((1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32)
    if end_latent is not None:
        logger.info(f"Use end image(s): {args.end_image_path}")
        history_latents[:, :, :1] = end_latent.to(history_latents)

    # prepare clean latents and indices
    start_latent = context_img[0]["start_latent"]
    history_latents = torch.cat([history_latents, start_latent], dim=2)
    total_generated_latent_frames = 1  # a bit hacky, but we employ the same logic as in official code
    latent_paddings = [0] * total_latent_sections  # dummy paddings for F1 mode

    latent_paddings = list(latent_paddings)  # make sure it's a list
    for loop_index in range(total_latent_sections):
        latent_padding = latent_paddings[loop_index]

        section_index = loop_index  # 0, 1, 2, 3
        section_index_from_last = section_index - total_latent_sections  # -4, -3, -2, -1
        is_last_section = loop_index == total_latent_sections - 1
        is_first_section = loop_index == 0
        latent_padding_size = 0  # dummy padding for F1 mode

        # select start latent
        if section_index_from_last in context_img:
            image_index = section_index_from_last
        elif section_index in context_img:
            image_index = section_index
        else:
            image_index = 0

        start_latent = context_img[image_index]["start_latent"]
        image_path = context_img[image_index]["image_path"]
        if image_index != 0:  # use section image other than section 0
            logger.info(
                f"Apply experimental section image, latent_padding_size = {latent_padding_size}, image_path = {image_path}"
            )

        # F1 mode
        indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
        (
            clean_latent_indices_start,
            clean_latent_4x_indices,
            clean_latent_2x_indices,
            clean_latent_1x_indices,
            latent_indices,
        ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

        clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]) :, :, :].split(
            [16, 2, 1], dim=2
        )
        clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

        # prepare conditioning inputs
        if section_index_from_last in context:
            prompt_index = section_index_from_last
        elif section_index in context:
            prompt_index = section_index
        else:
            prompt_index = 0

        context_for_index = context[prompt_index]
        # if args.section_prompts is not None:
        logger.info(f"Section {section_index}: {context_for_index['prompt']}")

        llama_vec = context_for_index["llama_vec"].to(device, dtype=torch.bfloat16)
        llama_attention_mask = context_for_index["llama_attention_mask"].to(device)
        clip_l_pooler = context_for_index["clip_l_pooler"].to(device, dtype=torch.bfloat16)

        image_encoder_last_hidden_state = context_img[image_index]["image_encoder_last_hidden_state"].to(
            device, dtype=torch.bfloat16
        )

        llama_vec_n = context_null["llama_vec"].to(device, dtype=torch.bfloat16)
        llama_attention_mask_n = context_null["llama_attention_mask"].to(device)
        clip_l_pooler_n = context_null["clip_l_pooler"].to(device, dtype=torch.bfloat16)

        preprocess_magcache(args, dit)

        generated_latents = sample_hunyuan(
            transformer=dit,
            sampler=args.sample_solver,
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=args.guidance_scale,
            distilled_guidance_scale=args.embedded_cfg_scale,
            guidance_rescale=args.guidance_rescale,
            shift=args.flow_shift,
            num_inference_steps=args.infer_steps,
            generator=seed_g,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=device,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
        )
        postprocess_magcache(args, dit)

        # concatenate generated latents
        total_generated_latent_frames += int(generated_latents.shape[2])
        # F1 mode: append generated latents to history latents
        history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
        real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

        logger.info(f"Generated. Latent shape {real_history_latents.shape}")
    
    # Only clean up shared models if they were created within this function
    wait_for_clean_memory = False
    if "dit" in locals():  # if model was loaded locally
        del dit
        synchronize_device(device)
        wait_for_clean_memory = True

    # wait for 5 seconds until block swap is done
    if wait_for_clean_memory and args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    return _save_latent(args, real_history_latents)


def _save_latent(args, real_history_latents):
    from musubi_tuner.fpack_generate_video import (
        save_latent,
    )
    from musubi_tuner.modules.custom_offloading_utils import clean_memory_on_device, synchronize_device

    latent = real_history_latents[0]

    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= 8
    width *= 8

    latent_path = save_latent(latent, args, height, width)

    del real_history_latents, latent
    synchronize_device(device)
    clean_memory_on_device(device)

    log = f"Save latent {latent_path}"
    print(log)
    q.put((True, log))
    return latent_path

def _decode_latent(args, latent_path: str):
    global q

    log = "Decode latent"
    print(log)
    q.put((False, log))

    import musubi_tuner
    from safetensors.torch import load_file
    from safetensors import safe_open

    vae = musubi_tuner.fpack_generate_video.load_vae(
        args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device
    )

    latent = load_file(latent_path)["latent"]
    with safe_open(latent_path, framework="pt") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    print(f"Loaded metadata: {metadata}")

    if "seeds" in metadata:
        seed = int(metadata["seeds"])
    if "height" in metadata and "width" in metadata:
        height = int(metadata["height"])
        width = int(metadata["width"])
        args.video_size = [height, width]
    if "video_seconds" in metadata:
        args.video_seconds = float(metadata["video_seconds"])

    total_latent_sections = (args.video_seconds * 30) / (args.latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    video = musubi_tuner.fpack_generate_video.decode_latent(
        args.latent_window_size,
        total_latent_sections,
        args.bulk_decode,
        vae,
        latent,
        device,
        args.one_frame_inference is not None
    )
    print("output_type", args.output_type)
    if args.output_type == "video" or args.output_type == "both":
        save_path = musubi_tuner.fpack_generate_video.save_video(video, args)
        log = f"Save video {save_path}"
    elif args.output_type == "images" or args.output_type == "latent_images":
        save_path = musubi_tuner.fpack_generate_video.save_images(video, args)
        log = f"Save images {save_path}"
        save_path = list(Path(save_path).glob("*.png"))

    print(log)
    q.put((True, log))
    return save_path

class FramePack:
    def __init__(self, manager):
        self.manager = manager

        self.fp_dit_fp8_path = MODEL_CACHE_DIR / "framepack_dit_fp8_scaled.safetensors"
        self.fp_dit_f1_fp8_path = MODEL_CACHE_DIR / "framepack_dit_f1_fp8_scaled.safetensors"
        self.fp_llm_fp8_path = MODEL_CACHE_DIR / "framepack_llm_fp8.safetensors"

        if self.manager:
            self.args = Args(
                save_path=str(get_save_path())
            )

            dataset = Dataset()
            self.dataset_config = DatasetConfig(
                datasets=[dataset]
            )

            self.precomputed_image_data = self.manager.dict({
                "height": None,
                "width": None,
                "video_seconds": None,
                "context_img": None,
                "end_latent": None,
                "control_mask_images": None,
                "control_latents": None,
            })
            self.precomputed_text_data = self.manager.dict({
                "context": None,
                "context_null": None,
            })
    
    def ns_to_dc(self, args):
        dc_fields = {f.name for f in fields(self.args)}
        updates = {
            name: getattr(args, name)
            for name in dc_fields
            if hasattr(args, name)
        }
        return updates

    def models_download_worker(self, pool: Pool, q: Queue):
        args = self.manager.Namespace(**asdict(self.args))
        p = pool.apply_async(_models_download, args=(args,))
        print("FramePack", "models_download_worker", "start")
        while not p.ready():
            try:
                isEnd, message = q.get(timeout=2)
                yield message
                if isEnd:
                    break
            except queue.Empty:
                if p.ready():
                    try:
                        p.get()
                    except Exception as e:
                        message = f"❌ worker failed: {e}"
                        print(message)
                        raise message
                    break
        p.get()
        updates = self.ns_to_dc(args)
        self.args = replace(self.args, **updates)
        print("FramePack", "models_download_worker", "end")
    
    def convert_worker(self, pool: Pool, q: Queue):
        print("FramePack", "convert_worker", "start")
        args = self.manager.Namespace(**asdict(self.args))
        p = pool.apply_async(_convert_llm_fp8, args=(self.fp_llm_fp8_path, args))
        print("FramePack", "_convert_llm_fp8", "start")
        while not p.ready():
            try:
                isEnd, message = q.get(timeout=2)
                yield message
                if isEnd:
                    break
            except queue.Empty:
                if p.ready():
                    try:
                        p.get()
                    except Exception as e:
                        message = f"❌ worker failed: {e}"
                        print(message)
                        raise message
                    break
        p.get()
        print("FramePack", "_convert_llm_fp8", "end")
        if self.args.f1:
                p = pool.apply_async(_convert_dit_f1_fp8_scaled, args=(self.fp_dit_f1_fp8_path, args))
                print("FramePack", "_convert_dit_f1_fp8_scaled", "start")
                while not p.ready():
                    try:
                        isEnd, message = q.get(timeout=2)
                        yield message
                        if isEnd:
                            break
                    except queue.Empty:
                        if p.ready():
                            try:
                                p.get()
                            except Exception as e:
                                message = f"❌ worker failed: {e}"
                                print(message)
                                raise message
                            break
                p.get()
        else:
            p = pool.apply_async(_convert_dit_fp8_scaled, args=(self.fp_dit_fp8_path, args))
            print("FramePack", "_convert_dit_fp8_scaled", "start")
            while not p.ready():
                try:
                    isEnd, message = q.get(timeout=2)
                    yield message
                    if isEnd:
                        break
                except queue.Empty:
                    if p.ready():
                        try:
                            p.get()
                        except Exception as e:
                            message = f"❌ worker failed: {e}"
                            print(message)
                            raise message
                        break
            p.get()
        updates = self.ns_to_dc(args)
        self.args = replace(self.args, **updates)
        print("FramePack", "_convert_dit_fp8_scaled or convert_dit_fp8_scaled", "end")
        print("FramePack", "convert_worker", "end")
    
    def prepare_i2v_inputs_worker(self, pool: Pool, q: Queue):
        args = self.manager.Namespace(**asdict(self.args))
        p = pool.apply_async(
            _prepare_i2v_inputs,
            args=(self.fp_llm_fp8_path, args, self.precomputed_image_data, self.precomputed_text_data)
        )
        print("FramePack", "prepare_i2v_inputs_worker", "start")
        while not p.ready():
            try:
                isEnd, message = q.get(timeout=2)
                yield message
                if isEnd:
                    break
            except queue.Empty:
                if p.ready():
                    try:
                        p.get()
                    except Exception as e:
                        message = f"❌ worker failed: {e}"
                        print(message)
                        raise message
                    break
        p.get()
        updates = self.ns_to_dc(args)
        self.args = replace(self.args, **updates)
        print("FramePack", "prepare_i2v_inputs_worker", "end")

    def one_frame_inference(self, pool: Pool, q: Queue):
        args = self.manager.Namespace(**asdict(self.args))
        p = pool.apply_async(
            _one_frame_inference,
            args=(self.fp_dit_fp8_path, args, self.precomputed_image_data, self.precomputed_text_data)
        )
        print("FramePack", "one_frame_inference", "start")
        while not p.ready():
            try:
                isEnd, message = q.get(timeout=2)
                yield message, None
                if isEnd:
                    break
            except queue.Empty:
                if p.ready():
                    try:
                        p.get()
                    except Exception as e:
                        message = f"❌ worker failed: {e}"
                        print(message)
                        raise message
                    break
        latent_path = p.get()
        updates = self.ns_to_dc(args)
        self.args = replace(self.args, **updates)
        yield None, latent_path
        print("FramePack", "one_frame_inference", "end")

    def generate_video_worker(self, pool: Pool, q: Queue):
        args = self.manager.Namespace(**asdict(self.args))
        if self.args.f1:
            p = pool.apply_async(
                _generate_f1_video,
                args=(self.fp_dit_f1_fp8_path, args, self.precomputed_image_data, self.precomputed_text_data)
            )
        else:
            p = pool.apply_async(
                _generate_video,
                args=(self.fp_dit_fp8_path, args, self.precomputed_image_data, self.precomputed_text_data)
            )
        print("FramePack", "generate_video_worker", "start", "f1", self.args.f1)
        while not p.ready():
            try:
                isEnd, message = q.get(timeout=2)
                yield message, None
                if isEnd:
                    break
            except queue.Empty:
                if p.ready():
                    try:
                        p.get()
                    except Exception as e:
                        message = f"❌ worker failed: {e}"
                        print(message)
                        raise message
                    break
        latent_path = p.get()
        updates = self.ns_to_dc(args)
        self.args = replace(self.args, **updates)
        yield None, latent_path
        print("FramePack", "generate_video_worker", "end")

    def decode_latent_worker(self, pool: Pool, q: Queue, latent_path):
        args = self.manager.Namespace(**asdict(self.args))
        p = pool.apply_async(
            _decode_latent,
            args=(args, latent_path)
        )
        print("FramePack", "decode_latent_worker", "start")
        while not p.ready():
            try:
                isEnd, message = q.get(timeout=2)
                yield message, None
                if isEnd:
                    break
            except queue.Empty:
                if p.ready():
                    try:
                        p.get()
                    except Exception as e:
                        message = f"❌ worker failed: {e}"
                        print(message)
                        raise message
                    break
        video_path = p.get()
        updates = self.ns_to_dc(args)
        self.args = replace(self.args, **updates)
        yield None, video_path
        print("FramePack", "decode_latent_worker", "end")

