Golden HF-skeleton fixes for OV2 30B-A3B p16m33 exports (applied by convert.sh do_fixup):
- modeling_llava_onevision2_moe.py: (1) _init_weights re-inits VisionRotaryEmbedding inv_freq_t/h/w
  (persistent=False buffers are garbage after meta-tensor from_pretrained; mirrors official 4B/Qwen3-VL);
  (2) prepare_inputs_for_generation gates pixel-drop on is_first_iteration (transformers 5.x removed
  cache_position for remote code); (3) convert_rope_to_block_layout_by_positions called with
  self.spatial_merge_size (was hardcoded 2; p16m33 is 3).
- chat_template.jinja: required by lmms-eval chat impl; auto_model skeleton may lack it.
config.json / preprocessor jsons are patched programmatically by do_fixup (pos_enc false; patch/merge/
temporal synced from vision_config). Verified 2026-07-05 against mcore ckpt + official 4B reference.

Qwen3.5 backbone (2026-09-07, the qwen35 s1.5 line; every item gated on text_config.model_type == qwen3_5*):
- modeling_llava_onevision2_moe.py: (4) M-RoPE -- when the text config declares rope_parameters.mrope_section the
  composite hands the text model [text; t; h; w] positions built by a torch-only port of the training-side
  get_rope_index + _grid_rows_per_vision_run (lmms-eval collapses video frames into one [[F,h,w]] grid row with F
  per-frame vision runs); prefill caches rope_deltas, decode steps add them. Without it HF's Qwen3.5 text model
  expands the 1D positions to identical t/h/w = plain 1D rope on image tokens (silently NOT the trained model).
  (5) _keys_to_ignore_on_load_unexpected = ^mtp. (Qwen3.5-native MTP layout exported by the bridge; never used at inference).
- configuration_llava_onevision2_moe.py: FALLBACK config class that dispatches text_config by model_type via
  CONFIG_MAPPING (LlavaConfig pattern). Installed by convert.sh fixup / build_qwen35_hf_skeleton.py ONLY when the
  skeleton's own class fails the dispatch self-test (hardcoded Qwen3MoeConfig -> silent coercion to qwen3_moe).
- convert.sh fixup (d): forces the golden chat_template (Qwen3.5-native adds <think> + drops the system line),
  strips tokenizer_config chat_template, sets image/video/vision_start/vision_end ids from tokenizer.json, tie=false,
  asserts vision out_hidden_size == text hidden_size and mrope_section present, writes Qwen3.5 eos/pad.
- Skeleton for CFG=: examples/models/qwen/qwen35_vl_ov2/convert/build_qwen35_hf_skeleton.py.
A800/convert/hf_skeleton_fixes is kept byte-identical to this directory (copy, not symlink).
