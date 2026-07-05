Golden HF-skeleton fixes for OV2 30B-A3B p16m33 exports (applied by convert.sh do_fixup):
- modeling_llava_onevision2_moe.py: (1) _init_weights re-inits VisionRotaryEmbedding inv_freq_t/h/w
  (persistent=False buffers are garbage after meta-tensor from_pretrained; mirrors official 4B/Qwen3-VL);
  (2) prepare_inputs_for_generation gates pixel-drop on is_first_iteration (transformers 5.x removed
  cache_position for remote code); (3) convert_rope_to_block_layout_by_positions called with
  self.spatial_merge_size (was hardcoded 2; p16m33 is 3).
- chat_template.jinja: required by lmms-eval chat impl; auto_model skeleton may lack it.
config.json / preprocessor jsons are patched programmatically by do_fixup (pos_enc false; patch/merge/
temporal synced from vision_config). Verified 2026-07-05 against mcore ckpt + official 4B reference.
