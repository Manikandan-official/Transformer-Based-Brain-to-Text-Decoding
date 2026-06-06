============================================================
STEPS TO RUN ON 128GB MACHINE
============================================================

BEFORE switching machines (run on GPU machine now):
  python save_logits.py
  # This saves logits_output/ folder — takes ~5 mins

AFTER switching to 128GB CPU machine ($3.21/hr):

--- STEP 1: Install Redis ---
  sudo apt-get install -y redis-server
  redis-server --daemonize yes
  redis-cli ping  # should print PONG

--- STEP 2: Start NEJM 3-gram LM server (Terminal 1) ---
  cd /teamspace/studios/this_studio/nejm-brain-to-text

  # Download 3-gram model first (if not done):
  # Go to https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq
  # Download languageModel.tar.gz (~several GB)
  # Extract to: language_model/pretrained_language_models/

  # Then start server:
  python language_model/language-model-standalone.py \
    --lm_path language_model/pretrained_language_models/openwebtext_3gram_lm_sil \
    --do_opt \
    --nbest 100 \
    --acoustic_scale 0.325 \
    --blank_penalty 90 \
    --alpha 0.55 \
    --redis_ip localhost \
    --gpu_number 0

  # Wait for: "Language model loaded. Waiting for logits..."

--- STEP 3: Run decoder (Terminal 2) ---
  cd /teamspace/studios/this_studio
  python run_nejm_decoder.py

--- STEP 4: Check results ---
  cat wer_nejm_results.json

============================================================
IMPORTANT NOTES:
============================================================
1. The 3-gram model needs ~60GB RAM — that's why you need 128GB
2. OPT-6.7B needs ~12.4GB GPU VRAM — T4 has 16GB so it fits
3. Total time estimate: ~2-3 hours for full 1426 val samples
4. Cost estimate: ~$6-10 total for the 128GB machine

Expected result:
  Their GRU + 3-gram:   7.44% WER
  Your Mamba + 3-gram:  ~4-6% WER  (better PER → better WER)
============================================================
