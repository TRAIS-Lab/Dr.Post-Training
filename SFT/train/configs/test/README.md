# Test Configs for Compression Separation

These configs test all combinations of score_grad_compression and opt_grad_compression.
Run with: `bash SFT/train/train.sh --methods <name> --train alpaca --task samsum --max_steps 3`

| Config | Score Compression | Update Compression | Expected Behavior |
|--------|------------------|--------------------|-------------------|
| test-score-only | normal-64*64 | none | Compressed scoring, full gradient updates |
| test-update-only | none | normal-512*512 | Full gradient scoring, MeSO compressed updates |
| test-both-shared | normal-512*512 | normal-512*512 | Same compressor shared for both |
| test-both-different | normal-64*64 | normal-512*512 | Separate compressors for score vs update |
| test-none | none | none | No compression at all (exact scoring, full updates) |
