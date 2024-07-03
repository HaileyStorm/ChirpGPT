# ~GPT2 (180.8M...181.5M if 32khz works out) with Native Audio Generation... and Birds!
Chirps in, chirps out. The idea is for the model to be able to take a short recording and respond with an appropriate (similar) birdsong. With luck.
# Inspired by https://github.com/nivibilla/build-nanogpt/tree/audio and Kate the Great.
# Data: https://huggingface.co/datasets/DBD-research-group/BirdSet
WIP, plan is to pull from XCL recordings with quality A, latitude between 33 and 51 and longitude between -125 to -85, at least 7 seconds long and with at least one entry in `detected_events`.

# To train...
- run `python sherlock.py`
- run `python train_gpt2.py`


# Samples

