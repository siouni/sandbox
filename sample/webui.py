
import os
from pathlib import Path
import tempfile

import gradio as gr

import torch.multiprocessing as mp

MODEL_CACHE_DIR = Path("./models")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

TEMP_PATH = Path("temp")
TEMP_PATH.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(TEMP_PATH)

from echo_studio.framepack import FramePack, FramePackUI
from echo_studio.gemini_cli import GeminiCLI, GeminiCLIUI
from echo_studio.extract_frames import ExtractFrames, ExtractFramesUI

if gr.NO_RELOAD:
    manager = None

def set_manager(manager_local):
    print(__name__, "set_manager", manager_local)
    global manager
    manager = manager_local

def get_manager():
    global manager
    print(__name__, "get_manager", manager)
    return manager

if gr.NO_RELOAD and __name__ == "__main__":
    print(__name__, "create manager")
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    set_manager(manager)

if not gr.NO_RELOAD:
    manager = get_manager()
    print(__name__, "reload", manager)

framepack = FramePack(manager)
gemini_cli = GeminiCLI(manager)
extract_frames = ExtractFrames(manager)

if __name__ != "__mp_main__":
    print(__name__, "framepack", framepack)
    framepack_ui = FramePackUI(framepack)
    gemini_cli_ui = GeminiCLIUI(gemini_cli)
    extract_frames_ui = ExtractFramesUI(extract_frames)

    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.Tab("FramePack") as framepack_tab:
                framepack_ui.inference_mode.render()
                framepack_ui.message.render()
                with gr.Row():
                    with gr.Column(scale=2):
                        with framepack_ui.inference_mode_tabs.render():
                            with framepack_ui.video_tab.render():
                                framepack_ui.f1_mode_enable.render()
                                framepack_ui.output_video_path.render()
                            with framepack_ui.one_frame_inference_tab.render():
                                framepack_ui.output_image_gallery.render()
                        framepack_ui.start_image_path.render()
                        framepack_ui.start_image_path_state.render()
                        with framepack_ui.advanced_video_control_panel.render():
                            framepack_ui.section_padding_enable.render()
                            for n in range(FramePackUI.SECTION_NUMBER_MAX):
                                with framepack_ui.section_areas[n].render():
                                    framepack_ui.section_start_image_path_states[n].render()
                                    framepack_ui.section_titles[n].render()
                                    framepack_ui.section_start_image_paths[n].render()
                                    framepack_ui.section_prompt_texts[n].render()
                                    framepack_ui.section_padding_number[n].render()
                        framepack_ui.end_image_path.render()
                        framepack_ui.latent_path.render()
                    with gr.Column(scale=2):
                        framepack_ui.prompt_text.render()
                        with gr.Row():
                            with gr.Column(scale=1, min_width=80):
                                framepack_ui.video_seconds_number.render()
                            with gr.Column(scale=1, min_width=80):
                                framepack_ui.video_sections_number.render()
                            with gr.Column(scale=1, min_width=80):
                                framepack_ui.seed_number.render()
                        with gr.Row():
                            with gr.Column(scale=2, min_width=80):
                                framepack_ui.size_width_number.render()
                            with gr.Column(scale=1, min_width=80):
                                framepack_ui.size_swap_btn.render()
                            with gr.Column(scale=2, min_width=80):
                                framepack_ui.size_height_number.render()
                    with gr.Column(scale=1):
                        with gr.Accordion("高度な設定", open=False):
                            framepack_ui.blocks_to_swap_number.render()
                            framepack_ui.vae_chunk_size_number.render()
                            framepack_ui.vae_spatial_tile_sample_min_size_number.render()
                            framepack_ui.one_frame_inference_text.render()
                        framepack_ui.generate_btn.render()
                        framepack_ui.decode_btn.render()
            with gr.Tab("Gemini CLI") as gemini_cli_tab:
                gemini_cli_ui.upload_file_state.render()
                gemini_cli_ui.chatbot.render()
                with gr.Row():
                    with gr.Column(scale=2):
                        gemini_cli_ui.system_prompt_text.render()
                        gemini_cli_ui.prompt_text.render()
                    with gr.Column(scale=1):
                        gemini_cli_ui.file_upload.render()
                    with gr.Column(scale=1):
                        gemini_cli_ui.send_btn.render()
            with gr.Tab("フレーム抽出") as extract_frames_tab:
                extract_frames_ui.message.render()
                with gr.Row():
                    with gr.Column(scale=2):
                        extract_frames_ui.input_video_path.render()
                    with gr.Column(scale=2):
                        extract_frames_ui.output_image_gallery.render()
                    with gr.Column(scale=1):
                        extract_frames_ui.extract_frames_btn.render()
            # with gr.Tab("LLM") as llm:
            #     pass

    framepack_ui.reg_events(demo)
    gemini_cli_ui.reg_events(demo)
    extract_frames_ui.reg_events(demo)

if __name__ == "__main__":
    print(__name__, "demo.launch")
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=8003,
        share=False,
        allowed_paths=[],
        inbrowser=False,
    )