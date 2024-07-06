# ~GPT2 (181.25M) with Native Audio Generation... and Birds!
Chirps in, chirps out. The idea is for the model to be able to take a short recording and respond with an appropriate (similar) birdsong. With luck.
# Inspired by https://github.com/nivibilla/build-nanogpt/tree/audio and Kate the Great.
# Data: https://huggingface.co/datasets/DBD-research-group/BirdSet
WIP, plan is to pull from XCL, HSN, SNE and possibly POW & SSW recordings with quality A or B, latitude between 33 and 51 and longitude between -125 to -85, at least 6 seconds long and with at least (length // 6 seconds) entries in `detected_events` (and ultimately at least one event per 6 second chunk kept).

# To train...
- run `python sherlock.py`
- run `python train_gpt2.py`


# Samples

