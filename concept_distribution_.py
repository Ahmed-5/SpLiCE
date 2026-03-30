import argparse
import torch
from torch.utils.data import DataLoader
import splice
import experiments.datasets as datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def run_concept_distribution(splicemodel, dataloader, vocab, dataset_name,
                              out_folder, class_label=None, plot_topk=10,
                              verbose=False, device="cuda"):
    """
    Decompose a dataset (or a single class) with SpLiCE, write sorted concept
    weights to weights.txt, and save a seaborn horizontal bar-plot PDF.

    Parameters
    ----------
    splicemodel  : loaded SpLiCE model (return_weights=True)
    dataloader   : DataLoader for the target dataset
    vocab        : list[str]  full concept vocabulary
    dataset_name : str        used for the plot title and file name
    out_folder   : str        directory where weights.txt and the PDF are saved
    class_label  : int|None   if set, decompose only that class index
    plot_topk    : int        how many top concepts to show in the bar plot
    verbose      : bool       print progress / stats to stdout
    device       : str        "cuda" or "cpu"

    Returns
    -------
    (concept_names, concept_weights, l0_norm, cosine)
      concept_names   : list[str]   all non-zero concept names, sorted by weight
      concept_weights : list[float] corresponding weights
      l0_norm         : float       average L0 sparsity over the dataset
      cosine          : float       average CLIP / SpLiCE cosine similarity
    """
    # ── decompose ────────────────────────────────────────────────────────────
    if class_label is None:
        if verbose:
            print("Decomposing " + str(dataset_name) + "...")
        weights, l0_norm, cosine = splice.decompose_dataset(dataloader, splicemodel, device)
    else:
        if verbose:
            print("Decomposing class " + str(class_label) + "...")
        class_weights_map, l0_norm, cosine = splice.decompose_classes(
            dataloader, class_label, splicemodel, device
        )
        weights = class_weights_map[class_label]

    _, indices = torch.sort(weights, descending=True)

    concept_names   = []
    concept_weights = []

    # ── save weights.txt ─────────────────────────────────────────────────────
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "weights.txt"), "w") as f:
        f.write("Concept Decomposition: \n")
        if verbose:
            print("Concept Decomposition:")

        for idx in indices.squeeze():
            w = weights[idx.item()].item()
            if w == 0:
                break
            line = "\t" + str(vocab[idx.item()]) + "\t" + str(round(w, 4))
            f.write(line + "\n")
            if verbose:
                print(line)
            concept_names.append(str(vocab[idx.item()]))
            concept_weights.append(w)

        if verbose:
            f.write("Average Decomposition L0 Norm: \t" + str(l0_norm) + "\n")
            print("Average Decomposition L0 Norm: \t" + str(l0_norm))
            f.write("Average CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)) + "\n")
            print("Average CLIP, SpLiCE Cosine Sim: \t" + str(round(cosine, 4)))

    # ── bar plot PDF ──────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "concept": concept_names[:plot_topk],
        "weight":  concept_weights[:plot_topk],
    })
    sns.set_style("darkgrid", {"axes.facecolor": "whitesmoke"})
    custom_palette = sns.color_palette(["#e86276ff", "#629d1eff"])
    sns.set_palette(custom_palette, 2)
    fig, ax = plt.subplots()
    sns.barplot(y="concept", x="weight", data=df, label="concept", orient="h", ax=ax)

    title = str(dataset_name)
    if class_label is not None:
        title += " Class " + str(class_label)
    title += " Decomposition"

    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Weight", fontsize=16)
    ax.set_ylabel("Concept", fontsize=16)
    ax.get_legend().remove()
    sns.despine(bottom=True)
    plt.tight_layout()
    titlepath = "_".join(title.split(" ")).lower()
    plt.savefig(os.path.join(out_folder, titlepath + ".pdf"))
    plt.close(fig)

    return concept_names, concept_weights, l0_norm, cosine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset",     type=str,   required=True)
    parser.add_argument("-data_path",   type=str)
    parser.add_argument("-out_folder",  type=str)
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("-l1_penalty",  type=float, default=0.2)
    parser.add_argument("-class_label", type=int)
    parser.add_argument("-device",      type=str,   default="cuda")
    parser.add_argument("-model",       type=str,   default="open_clip:ViT-B-32")
    parser.add_argument("-vocab",       type=str,   default="laion")
    parser.add_argument("-vocab_size",  type=int,   default=10000)
    parser.add_argument("-batch_size",  type=int,   default=512)
    parser.add_argument("-plot_topk",   type=int,   default=10)
    args = parser.parse_args()

    # ── build objects from CLI args, then delegate to run_concept_distribution
    preprocess  = splice.get_preprocess(args.model)
    dataset     = datasets.load(args.dataset, preprocess, args.data_path)
    dataloader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    splicemodel = splice.load(
        args.model, args.vocab, args.vocab_size, args.device,
        l1_penalty=args.l1_penalty, return_weights=True,
    )
    vocab = splice.get_vocabulary(args.vocab, args.vocab_size)

    run_concept_distribution(
        splicemodel=splicemodel,
        dataloader=dataloader,
        vocab=vocab,
        dataset_name=args.dataset,
        out_folder=args.out_folder,
        class_label=args.class_label,
        plot_topk=args.plot_topk,
        verbose=args.verbose,
        device=args.device,
    )


if __name__ == "__main__":
    main()
