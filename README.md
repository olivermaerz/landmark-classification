Udacity *Landmark Classification* project: PyTorch CNNs (from-scratch + transfer learning) to classify landmarks in images and serve top-k predictions in an app.

Setup with [uv](https://docs.astral.sh/uv/): `uv venv && source venv/bin/activate && uv pip install -r requirements.txt`

Then start JupyterLab with: `jupyter lab` and complete the notebooks in order: `cnn_from_scratch.ipynb`, `transfer_learning.ipynb`, then `app.ipynb`.

The landmark images are not included (size; subset of Google Landmarks Dataset v2). Place the train/test folders under `landmark_images/` locally before running.
