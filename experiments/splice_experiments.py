"""
splice_experiments.py
---------------------
All SpLiCE experiment scripts reimplemented as callable functions.
Each function takes explicit parameters instead of argparse, so they can be
called directly from splice_metrics_extended.py or any other driver.

Covered scripts
---------------
  zero_shot.py            -> run_zero_shot(), generate_label_embeddings(), find_closest()
  decompose_data.py       -> run_decompose_data()
  decompose_image.py      -> run_decompose_image()
  concept_distribution.py -> run_concept_distribution()
  concept_histogram.py    -> run_concept_histogram()
  embed_mscoco.py         -> run_embed_mscoco_images(), run_embed_mscoco_text()
  retrieval.py            -> run_retrieval(), CLIPDataset
  intervention.py         -> run_intervention_celeba(), run_intervention_waterbirds()
"""

import os
import json
import numpy as np
import torch
import splice
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model


# ══════════════════════════════════════════════════════════════════════════════
# zero_shot.py
# ══════════════════════════════════════════════════════════════════════════════

def find_closest(embedding, label_embeddings):
    """Return argmax dot-product class index for each embedding row."""
    return torch.argmax(embedding @ label_embeddings.T, dim=-1)


def generate_label_embeddings(dataset, splicemodel, tokenizer, device):
    """
    Build a (num_classes, dim) tensor of 'A photo of a {class}' text embeddings,
    L2-normalised, ready for dot-product classification.
    """
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    label_embeddings = [
        splicemodel.encode_text(tokenizer(f"A photo of a {idx_to_class[k]}").to(device))
        for k in sorted(idx_to_class)
    ]
    label_embeddings = torch.stack(label_embeddings).squeeze()
    label_embeddings /= torch.linalg.norm(label_embeddings, dim=-1).view(-1, 1)
    return label_embeddings


def run_zero_shot(splicemodel, dataloader, label_embeddings, device):
    """
    Zero-shot evaluation over a classification dataloader.

    Parameters
    ----------
    splicemodel       : loaded SpLiCE model (return_weights=True)
    dataloader        : DataLoader yielding (images, labels)
    label_embeddings  : (num_classes, dim) tensor from generate_label_embeddings()
    device            : "cuda" | "cpu"

    Returns
    -------
    (avg_accuracy, avg_l0_sparsity, avg_cosine_similarity)
    """
    total = correct = l0 = cosine = 0
    splicemodel.eval()
    for image, label in tqdm(dataloader, desc="zero-shot"):
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            original_emb = splicemodel.clip.encode_image(image)
            weights      = splicemodel.encode_image(image)
            embedding    = splicemodel.recompose_image(weights)
            cos_mat = (
                torch.nn.functional.normalize(embedding, dim=1)
                @ torch.nn.functional.normalize(original_emb, dim=1).T
            )
            preds    = find_closest(embedding, label_embeddings)
            cosine  += torch.sum(torch.diag(cos_mat)).item()
            l0      += torch.sum(torch.linalg.norm(weights, ord=0, dim=1)).item()
            correct += torch.sum(preds == label).item()
            total   += image.shape[0]
    return correct / total, l0 / total, cosine / total


# ══════════════════════════════════════════════════════════════════════════════
# decompose_data.py
# ══════════════════════════════════════════════════════════════════════════════

def run_decompose_data(splicemodel, dataloader, vocab, out_path,
                       class_label=None, verbose=False, device="cuda"):
    """
    Decompose a full dataset (or a single class) with SpLiCE and write the
    sorted concept weights to out_path.

    Parameters
    ----------
    splicemodel : SpLiCE model
    dataloader  : DataLoader for the target dataset
    vocab       : list[str]  concept vocabulary
    out_path    : str  path to the output .txt file
    class_label : int | None  if set, decompose only that class
    verbose     : bool  print to stdout as well
    device      : str

    Returns
    -------
    (weights, l0_norm, cosine)
    """
    if class_label is None:
        if verbose:
            print("Decomposing dataset...")
        weights, l0_norm, cosine = splice.decompose_dataset(dataloader, splicemodel, device)
    else:
        if verbose:
            print(f"Decomposing class {class_label}...")
        class_weights_map, l0_norm, cosine = splice.decompose_classes(
            dataloader, class_label, splicemodel, device
        )
        weights = class_weights_map[class_label]

    _, indices = torch.sort(weights, descending=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("Concept Decomposition:\n")
        for idx in indices.squeeze():
            w = round(weights[idx.item()].item(), 4)
            if str(w) == "0.0":
                break
            line = f"\t{vocab[idx.item()]}\t{w}"
            f.write(line + "\n")
            if verbose:
                print(line)
        f.write(f"Average Decomposition L0 Norm: \t{l0_norm}\n")
        f.write(f"Average CLIP, SpLiCE Cosine Sim: \t{round(cosine, 4)}\n")
        if verbose:
            print(f"Average Decomposition L0 Norm: \t{l0_norm}")
            print(f"Average CLIP, SpLiCE Cosine Sim: \t{round(cosine, 4)}")

    return weights, l0_norm, cosine


# ══════════════════════════════════════════════════════════════════════════════
# decompose_image.py
# ══════════════════════════════════════════════════════════════════════════════

def run_decompose_image(image_path, splicemodel, preprocess, vocab,
                        out_dir, verbose=False, device="cuda"):
    """
    Decompose a single image and write its concept weights to
    out_dir/<stem>_weights.txt.

    Returns
    -------
    (weights, l0_norm, cosine)
    """
    img = preprocess(Image.open(image_path)).to(device).unsqueeze(0)
    weights, l0_norm, cosine = splice.decompose_image(img, splicemodel, device)
    _, indices = torch.sort(weights, descending=True)

    stem     = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, stem + "_weights.txt")
    os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"Concept Decomposition of {image_path}:\n")
        if verbose:
            print(f"Concept Decomposition of {image_path}:")
        for idx in indices.squeeze():
            w = weights[0, idx.item()].item()
            if w == 0:
                break
            line = f"\t{vocab[idx.item()]}\t{round(w, 4)}"
            f.write(line + "\n")
            if verbose:
                print(line)
        f.write(f"Decomposition L0 Norm: \t{l0_norm}\n")
        f.write(f"CLIP, SpLiCE Cosine Sim: \t{round(cosine, 4)}\n")
        if verbose:
            print(f"Decomposition L0 Norm: \t{l0_norm}")
            print(f"CLIP, SpLiCE Cosine Sim: \t{round(cosine, 4)}")

    return weights, l0_norm, cosine


# ══════════════════════════════════════════════════════════════════════════════
# concept_distribution.py
# ══════════════════════════════════════════════════════════════════════════════

def run_concept_distribution(splicemodel, dataloader, dataset_name, vocab,
                              out_folder, class_label=None, plot_topk=10,
                              verbose=False, device="cuda"):
    """
    Decompose a dataset (or class), save sorted weights to weights.txt, and
    save a seaborn horizontal bar-plot PDF of the top-k concepts.

    Returns
    -------
    (concept_names, concept_weight_vals, l0_norm, cosine)
    """
    if class_label is None:
        weights, l0_norm, cosine = splice.decompose_dataset(dataloader, splicemodel, device)
    else:
        class_weights_map, l0_norm, cosine = splice.decompose_classes(
            dataloader, class_label, splicemodel, device
        )
        weights = class_weights_map[class_label]

    _, indices   = torch.sort(weights, descending=True)
    concept_names, concept_vals = [], []

    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "weights.txt"), "w") as f:
        f.write("Concept Decomposition:\n")
        for idx in indices.squeeze():
            w = weights[idx.item()].item()
            if w == 0:
                break
            f.write(f"\t{vocab[idx.item()]}\t{round(w, 4)}\n")
            if verbose:
                print(f"\t{vocab[idx.item()]}\t{round(w, 4)}")
            concept_names.append(str(vocab[idx.item()]))
            concept_vals.append(w)
        if verbose:
            f.write(f"Average Decomposition L0 Norm: \t{l0_norm}\n")
            f.write(f"Average CLIP, SpLiCE Cosine Sim: \t{round(cosine, 4)}\n")

    df  = pd.DataFrame({"concept": concept_names[:plot_topk], "weight": concept_vals[:plot_topk]})
    sns.set_style("darkgrid", {"axes.facecolor": "whitesmoke"})
    sns.set_palette(sns.color_palette(["#e86276ff", "#629d1eff"]), 2)
    fig, ax = plt.subplots()
    sns.barplot(y="concept", x="weight", data=df, label="concept", orient="h", ax=ax)

    title = f"{dataset_name}{(' Class ' + str(class_label)) if class_label else ''} Decomposition"
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Weight", fontsize=16)
    ax.set_ylabel("Concept", fontsize=16)
    ax.get_legend().remove()
    sns.despine(bottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "_".join(title.split()).lower() + ".pdf"))
    plt.close(fig)

    return concept_names, concept_vals, l0_norm, cosine


# ══════════════════════════════════════════════════════════════════════════════
# concept_histogram.py
# ══════════════════════════════════════════════════════════════════════════════

def _collect_concept_weights(splicemodel, dataloader, class_int_indices,
                              concept_vocab_indices, device):
    """
    Internal helper: collect per-sample concept weights for specified classes.
    Returns (class_weights_dict, avg_l0, avg_cosine).
    """
    splicemodel.eval()
    splicemodel.return_weights = True
    splicemodel.return_cosine  = True
    cw, ct = {}, {}
    l0_total = cosine_total = total = 0

    for _, (image, label) in enumerate(dataloader):
        mask = torch.zeros(label.shape[0])
        for ci in class_int_indices:
            mask = torch.logical_or((label == ci).to(torch.int64), mask)
        sel = torch.argwhere(mask >= 0).squeeze()
        if sel.nelement() == 0:
            continue
        image, label = image[sel], label[sel]
        if sel.nelement() == 1:
            image, label = image.unsqueeze(0), label.unsqueeze(0)
        with torch.no_grad():
            image, label = image.to(device), label.to(device)
            weights, batch_cosine = splicemodel.encode_image(image)
            target_w = weights[:, concept_vocab_indices].tolist()
            for i in range(image.shape[0]):
                li = label[i].item()
                if li in cw:
                    cw[li].append(target_w[i])
                    ct[li] += 1
                elif li in class_int_indices:
                    cw[li] = [target_w[i]]
                    ct[li] = 1
            l0_total     += torch.linalg.vector_norm(weights, dim=1, ord=0).sum().item()
            cosine_total += batch_cosine.item()
            total        += image.shape[0]

    return cw, l0_total / max(total, 1), cosine_total / max(total, 1)


def run_concept_histogram(splicemodel, dataloader_train, dataloader_test,
                          dataset_name, class_names, class_int_indices,
                          concepts, vocab, out_path, device="cuda"):
    """
    Compute per-class concept weight distributions (train + test combined)
    and save a log-scale seaborn histogram PDF.

    Parameters
    ----------
    splicemodel        : SpLiCE model
    dataloader_train   : training DataLoader
    dataloader_test    : test DataLoader
    dataset_name       : str  used in plot title
    class_names        : list[str]  human-readable class names (e.g. ['man','woman'])
    class_int_indices  : list[int]  integer label IDs matching class_names
    concepts           : list[str]  concept strings to track (must be in vocab)
    vocab              : list[str]  full concept vocabulary
    out_path           : str  directory for output PDF
    device             : str

    Returns
    -------
    (flat_classes, flat_weights, avg_l0, avg_cosine)
    """
    concept_vocab_indices = [list(vocab).index(c) for c in concepts]
    flat_classes, flat_weights = [], []
    last_l0 = last_cosine = 0.0

    for dl in [dataloader_train, dataloader_test]:
        cw, l0, cosine = _collect_concept_weights(
            splicemodel, dl, class_int_indices, concept_vocab_indices, device
        )
        last_l0, last_cosine = l0, cosine
        for class_int in cw.keys():
            weightlist = torch.mean(torch.tensor(cw[class_int]), dim=-1).tolist()
            flat_weights += weightlist
            name = class_names[class_int_indices.index(class_int)]
            flat_classes += [name] * len(weightlist)

    df = pd.DataFrame({"class": flat_classes, "weight": flat_weights})
    sns.set_style("darkgrid", {"axes.facecolor": "whitesmoke"})
    sns.set_palette(sns.color_palette(["#629d1eff", "#e86276ff"]), 2)
    fig, ax = plt.subplots()
    sns.histplot(x="weight", data=df, hue="class", bins=20, ax=ax)

    title = f"{dataset_name} Histogram"
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Weight", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.set_yscale("log")
    sns.despine(bottom=True)
    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(os.path.join(out_path, "_".join(title.split()).lower() + ".pdf"))
    plt.close(fig)

    return flat_classes, flat_weights, last_l0, last_cosine


# ══════════════════════════════════════════════════════════════════════════════
# embed_mscoco.py
# ══════════════════════════════════════════════════════════════════════════════

def _mscoco_data_dict(data_path):
    with open(os.path.join(data_path, "annotations/captions_train2017.json")) as f:
        data = json.load(f)
    img_id_set = {}
    for x in data["annotations"]:
        img_id_set.setdefault(x["image_id"], []).append(x["caption"])
    return img_id_set


def run_embed_mscoco_images(clip_model, preprocess, data_path, out_path,
                             batch_size=512, device="cuda"):
    """
    Encode all MS-COCO train images with clip_model and save each as a
    sparse .pth file in out_path.
    """
    img_id_set = _mscoco_data_dict(data_path)
    img_ids    = list(img_id_set.keys())
    os.makedirs(out_path, exist_ok=True)

    for i in tqdm(range(0, len(img_ids), batch_size), desc="embed images"):
        batch_ids = img_ids[i: i + batch_size]
        imgs = torch.zeros((len(batch_ids), 3, 224, 224)).to(device)
        for j, bid in enumerate(batch_ids):
            img_file = os.path.join(data_path, "train2017", str(bid).zfill(12) + ".jpg")
            imgs[j]  = preprocess(Image.open(img_file)).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(imgs).to_sparse_csr()
        for j, bid in enumerate(batch_ids):
            torch.save(feats[j], os.path.join(out_path, str(bid).zfill(12) + "_image.pth"))


def run_embed_mscoco_text(clip_model, tokenizer, data_path, out_path, device="cuda"):
    """
    Encode all MS-COCO captions with clip_model and save each image's
    caption tensor as a .pth file in out_path.
    """
    caption_set = _mscoco_data_dict(data_path)
    os.makedirs(out_path, exist_ok=True)

    for iid, captions in tqdm(caption_set.items(), desc="embed text"):
        text = tokenizer(captions).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = clip_model.encode_text(text)
        torch.save(feats, os.path.join(out_path, str(iid) + "_caption.pth"))


# ══════════════════════════════════════════════════════════════════════════════
# retrieval.py
# ══════════════════════════════════════════════════════════════════════════════

class CLIPDataset(torch.utils.data.Dataset):
    """Loads pre-embedded MS-COCO (image, caption) pairs from disk."""

    def __init__(self, ids, data_folder):
        self.ids         = ids
        self.data_folder = data_folder

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        iid      = self.ids[idx]
        data     = torch.load(os.path.join(self.data_folder, f"{iid.zfill(12)}.pth"))
        image    = data[0]
        captions = data[1:]
        caption  = captions[torch.randperm(captions.shape[0])[0]]
        return image, caption


def run_retrieval(splicemodel, mscoco_ids_file, embedded_data_path,
                  batch_size=1024, device="cuda", use_clip=False,
                  max_batches=9):
    """
    Image-text retrieval evaluation on pre-embedded MS-COCO data.

    Parameters
    ----------
    splicemodel         : SpLiCE model
    mscoco_ids_file     : str  path to text file with one MS-COCO image-id per line
    embedded_data_path  : str  directory containing *_image.pth / *_caption.pth files
    batch_size          : int
    device              : str
    use_clip            : bool  if True, skip SpLiCE re-encoding (use raw CLIP features)
    max_batches         : int  number of batches to evaluate (default 9 mirrors original)

    Returns
    -------
    dict with keys: loss_mean, loss_std,
                    img_r1_mean/std, txt_r1_mean/std,
                    img_r5_mean/std, txt_r5_mean/std,
                    img_r10_mean/std, txt_r10_mean/std
    """
    logit_scale = splicemodel.clip.logit_scale
    if not use_clip:
        splicemodel.clip = None

    with open(mscoco_ids_file) as f:
        lines = [l.rstrip() for l in f]
    dataloader = DataLoader(CLIPDataset(lines, embedded_data_path),
                            batch_size=batch_size, shuffle=True)

    acc = {k: [] for k in ["loss", "img_r1", "txt_r1", "img_r5", "txt_r5", "img_r10", "txt_r10"]}

    for batch_i, (batch_x, batch_y) in enumerate(tqdm(dataloader, desc="retrieval")):
        with torch.no_grad():
            img_f = batch_x.to(device).float()
            txt_f = batch_y.to(device).float()
            img_f /= img_f.norm(dim=-1, keepdim=True)
            txt_f /= txt_f.norm(dim=-1, keepdim=True)
            if not use_clip:
                img_f = splicemodel.encode_image(img_f)
                img_f /= img_f.norm(dim=-1, keepdim=True)

            logits_img = logit_scale * img_f @ txt_f.t()
            logits_txt = logits_img.t()
            img_sort   = torch.argsort(logits_img, dim=1)
            txt_sort   = torch.argsort(logits_txt, dim=1)
            bs         = batch_x.shape[0]
            labels     = torch.arange(bs, device=device).long()

            for k, tag in [(1, "r1"), (5, "r5"), (10, "r10")]:
                acc[f"img_{tag}"].append(
                    torch.tensor([l in img_sort[l, -k:] for l in range(bs)]).float().mean()
                )
                acc[f"txt_{tag}"].append(
                    torch.tensor([l in txt_sort[l, -k:] for l in range(bs)]).float().mean()
                )
            loss = (
                torch.nn.functional.cross_entropy(logits_img, labels) +
                torch.nn.functional.cross_entropy(logits_txt, labels)
            ) / 2
            acc["loss"].append(loss.cpu())

        if batch_i + 1 >= max_batches:
            break

    results = {}
    for k, vals in acc.items():
        t = torch.stack(vals)
        results[f"{k}_mean"] = t.mean().item()
        results[f"{k}_std"]  = t.std().item()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# intervention.py  (internal helpers + public run_* functions)
# ══════════════════════════════════════════════════════════════════════════════

def _zs_eval_intervention(model, dataloader, label_embeddings, obj, device,
                           intervene=None):
    total = correct = 0
    class_0_acc = {"c": 0, "t": 0}
    class_1_acc = {"c": 0, "t": 0}
    model.eval()
    for _, (image, objlabel, adjlabel) in enumerate(dataloader):
        with torch.no_grad():
            image = image.to(device)
            label = (objlabel if obj else adjlabel).to(device)
            if intervene is None:
                embedding = model.encode_image(image)
            else:
                embedding = model.intervene_image(image, intervene).to(device)
            if embedding.shape[1] == 512:
                embedding /= torch.linalg.norm(embedding, dim=-1).view(-1, 1)
            preds    = find_closest(embedding, label_embeddings)
            correct += torch.sum((preds == label).to(torch.int64)).item()
            total   += image.shape[0]
            for i, y in enumerate(label):
                bucket = class_1_acc if y == 1 else class_0_acc
                bucket["t"] += 1
                if preds[i] == y:
                    bucket["c"] += 1
    return (correct / total,
            class_0_acc["c"] / max(class_0_acc["t"], 1),
            class_1_acc["c"] / max(class_1_acc["t"], 1))


def _concept_zs_eval(tokenizer, model, loader, labels, obj, device, intervene=None):
    embs = torch.stack(
        [model.encode_text(tokenizer(l).to(device)) for l in labels]
    ).squeeze()
    embs /= torch.linalg.norm(embs, dim=-1).view(-1, 1)
    a, a0, a1 = _zs_eval_intervention(model, loader, embs, obj, device, intervene)
    return round(a, 3), round(a0, 3), round(a1, 3)


def _train_cbm(X, y, reg):
    m = linear_model.LogisticRegression(
        penalty="l1", C=reg, solver="saga", fit_intercept=False
    )
    m.fit(X, y)
    return m


def _test_cbm(cbm, X, y):
    i0 = np.nonzero((y == 0).astype(np.int64))
    i1 = np.nonzero((y == 1).astype(np.int64))
    return (cbm.score(X, y),
            cbm.score(X[i0], y[i0]),
            cbm.score(X[i1], y[i1]))


def _test_cbm_subgroup(cbm, X, y1, y2):
    i00 = np.nonzero(np.logical_and(y1 == 0, y2 == 0).astype(np.int64))
    i10 = np.nonzero(np.logical_and(y1 == 1, y2 == 0).astype(np.int64))
    i01 = np.nonzero(np.logical_and(y1 == 0, y2 == 1).astype(np.int64))
    i11 = np.nonzero(np.logical_and(y1 == 1, y2 == 1).astype(np.int64))
    return (cbm.score(X, y1),
            cbm.score(X[i00], y1[i00]),
            cbm.score(X[i10], y1[i10]),
            cbm.score(X[i01], y1[i01]),
            cbm.score(X[i11], y1[i11]))


def _collect_features(splicemodel, dataloader, device):
    splicemodel.return_weights = True
    X = Y1 = Y2 = None
    for imgs, y1, y2 in dataloader:
        with torch.no_grad():
            feats = splicemodel.encode_image(imgs.to(device))
        X  = feats if X  is None else torch.cat((X,  feats))
        Y1 = y1   if Y1 is None else torch.cat((Y1, y1))
        Y2 = y2   if Y2 is None else torch.cat((Y2, y2))
    return X.cpu().numpy(), Y1.cpu().numpy(), Y2.cpu().numpy()


def run_intervention_celeba(splicemodel, preprocess, tokenizer, vocab,
                             intervening_indices, data_path,
                             probe_regularization=1.0, device="cuda"):
    """
    Zero-shot + linear-probe + intervention analysis on CelebA.

    Returns
    -------
    dict with keys:
      clip_gender_zs, clip_glasses_zs,
      splice_gender_zs, splice_glasses_zs,
      intervened_gender_zs, intervened_glasses_zs,
      probe_gender, probe_glasses,
      probe_gender_intervened, probe_glasses_intervened
    Each value is a (full_acc, class0_acc, class1_acc) tuple.
    """
    from SpLiCE.experiments.datasets import CelebA
    ds_train     = CelebA(data_path, train=False, transform=preprocess)
    ds_test      = CelebA(data_path, train=True,  transform=preprocess)
    train_loader = DataLoader(ds_train, batch_size=1024, shuffle=True)
    test_loader  = DataLoader(ds_test,  batch_size=1024, shuffle=False)

    obj_labels = ["A picture of a man", "A picture of a woman"]
    adj_labels = ["A picture of person without glasses",
                  "A picture of person with glasses"]

    res = {}
    res["clip_gender_zs"]        = _concept_zs_eval(tokenizer, splicemodel.clip, test_loader, obj_labels, obj=True,  device=device)
    res["clip_glasses_zs"]       = _concept_zs_eval(tokenizer, splicemodel.clip, test_loader, adj_labels, obj=False, device=device)
    res["splice_gender_zs"]      = _concept_zs_eval(tokenizer, splicemodel,      test_loader, obj_labels, obj=True,  device=device)
    res["splice_glasses_zs"]     = _concept_zs_eval(tokenizer, splicemodel,      test_loader, adj_labels, obj=False, device=device)
    res["intervened_gender_zs"]  = _concept_zs_eval(tokenizer, splicemodel,      test_loader, obj_labels, obj=True,  device=device, intervene=intervening_indices)
    res["intervened_glasses_zs"] = _concept_zs_eval(tokenizer, splicemodel,      test_loader, adj_labels, obj=False, device=device, intervene=intervening_indices)

    X_tr, y_tr1, y_tr2 = _collect_features(splicemodel, train_loader, device)
    X_te, y_te1, y_te2 = _collect_features(splicemodel, test_loader,  device)

    cbm_g  = _train_cbm(X_tr, y_tr1, probe_regularization)
    cbm_gl = _train_cbm(X_tr, y_tr2, probe_regularization)

    res["probe_gender"]  = _test_cbm(cbm_g,  X_te, y_te1)
    res["probe_glasses"] = _test_cbm(cbm_gl, X_te, y_te2)

    for idx in intervening_indices:
        cbm_g.coef_[0, idx]  = 0
        cbm_gl.coef_[0, idx] = 0

    res["probe_gender_intervened"]  = _test_cbm(cbm_g,  X_te, y_te1)
    res["probe_glasses_intervened"] = _test_cbm(cbm_gl, X_te, y_te2)
    return res


def run_intervention_waterbirds(splicemodel, preprocess, vocab,
                                 intervening_indices, data_path,
                                 probe_regularization=1.0, device="cuda"):
    """
    Linear-probe + intervention analysis on Waterbirds.

    Returns
    -------
    dict with keys:
      probe_before  : (full, LL, WL, LW, WW) subgroup accuracies
      probe_after   : same after zeroing intervening_indices
    """
    from SpLiCE.experiments.datasets import WaterbirdDataset
    ds_train     = WaterbirdDataset(data_path, preprocess, train=True)
    ds_test      = WaterbirdDataset(data_path, preprocess, train=False)
    train_loader = DataLoader(ds_train, batch_size=512, shuffle=True)
    test_loader  = DataLoader(ds_test,  batch_size=512, shuffle=True)

    X_tr, y_tr1, _     = _collect_features(splicemodel, train_loader, device)
    X_te, y_te1, y_te2 = _collect_features(splicemodel, test_loader,  device)

    cbm = _train_cbm(X_tr, y_tr1, probe_regularization)
    res = {"probe_before": _test_cbm_subgroup(cbm, X_te, y_te1, y_te2)}

    for idx in intervening_indices:
        cbm.coef_[0, idx] = 0
    res["probe_after"] = _test_cbm_subgroup(cbm, X_te, y_te1, y_te2)
    return res
