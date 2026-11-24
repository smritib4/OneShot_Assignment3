# One-shot scene classification with SigLIP

ITCS 6169/8169 Computer Vision, Assignment 3.
Author: Smriti Bhemireddy

Question: how accurately can the 16 scene classes be recognized when each class contributes one labeled training image?

The dataset is the same 2,400 / 400 split used in Assignments 1 and 2. Training is allowed to open 16 images, one from each class. The other 2,384 training images stay closed. Test labels are used only to report accuracy.

## Method

The encoder is frozen [SigLIP SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) (`google/siglip-so400m-patch14-384`). Its weights are the public Hugging Face checkpoint. They are not fine-tuned.

Each class is turned into two unit vectors:

- A text vector. Four prompts ("a photo of a {phrase}.", and three variants) are embedded and averaged. The phrase is a fixed wording of the folder name, stored in `src/oneshot/labels.py`.
- A visual vector. The one support photo is embedded twice, as itself and as a horizontal flip, and those two vectors are averaged. The flip is the test-time augmentation, and the same two views are used for every test image.

A test image is assigned to the class with the highest scaled cosine similarity. Three classifiers are scored in the same pass:

- Text only. The class weight is the averaged prompt embedding. This is the submitted model. The first test pass scored it at 85.00% (340 of 400).
- One photo only. The class weight is the support image. That pass scored 68.75% (275 of 400).
- Equal mix. The class weight is the normalized sum of the text vector and the photo vector. That pass scored 82.00% (328 of 400).

The photo is a weaker class vector than the class name, and averaging the two lowered the text classifier by 3 points. The saved weights in `cache.pt` are the text vectors. The one labeled image is still embedded, and the other two accuracies are printed beside the official number. No extra training label is used, and the mix weight is not searched.

The 16 support file names are fixed in `supports/seed0.json`. They were drawn with `numpy.random.default_rng(0)` from the sorted file names in each class folder.

The saved training result is `runs/official/cache.pt` (the three sets of vectors) plus `runs/official/cache_meta.json`. `predict.py` loads that cache and does not read the training folders.

## Reproduce the official accuracy

Use a GPU. The command below is the reproduction. It rebuilds the cache from the 16 support images and prints one official test accuracy.

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --src /path/to/dataset
python evaluate.py --train-root data/train --test-root data/test --output-dir runs/official
```

`prepare_data.py` also accepts `--zip dataset_scenes.zip`. The zip must contain `train/` and `test/` with 150 and 25 images in each of the 16 class folders.

The line that belongs in the report looks like this:

```text
Official test accuracy: 00.00% (0/400)
```

`metrics.json` in the output directory stores that fraction, the text-only accuracy, the prototype-only accuracy, and the per-class counts. Those two extra accuracies are observations about the recipe. The submitted number is `official_test_accuracy`.

On a machine that already has the Assignment 2 copy of the data:

```bash
python evaluate.py --train-root C:/Users/smrit/SSL_Assignment2/data/train --test-root C:/Users/smrit/SSL_Assignment2/data/test
```

If the GPU runs out of memory, rerun with `--batch-size 4`.

Inference from a cache that is already in the repo:

```bash
python predict.py --cache-dir runs/official --image-root data/test
```

## Colab

This PC has no GPU. The matching notebook is `notebooks/Assignment3_Colab.ipynb`.

1. Create a public GitHub repo, push this project, and make sure the default branch is `main`.
2. Upload `dataset_scenes.zip` (about 55 MB) to the top of Google Drive. The zip is built locally and is not part of the git repo.
3. Open https://colab.research.google.com, set the runtime to a T4 GPU, and upload the notebook or open it from the repo.
4. Run the notebook from the top. The last cell prints the official accuracy.
5. Download `runs/official/` and commit `cache.pt`, `cache_meta.json`, `metrics.json`, `predictions.csv`, and `confusion.csv` so the repo contains the trained weights.

The dataset folder on Drive, if you need to rebuild the zip, is the same one as Assignments 1 and 2: https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M
