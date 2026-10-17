# References and verification

REF-001 through REF-012 retain source checks recorded on 2026-10-17. Additional
checks for the refined issue #4 protocol are dated separately below. Verification
is limited to the stated claim: an accessible card or build table does not establish
dataset integrity, image rights, compatibility of every pinned wheel, or reproduced
model performance. Partial verification and unresolved provenance remain explicit.

| ID | Source | Verification and limits |
|---|---|---|
| REF-001 | [Fashion Product Images Small](https://huggingface.co/datasets/ashraq/fashion-product-images-small) | Verified card and earlier Dataset Viewer metadata: 44,072 declared rows, one split, fields, total file size; inspected three image-dimension records. The local row count, images, checksums, and download revision are not validated. |
| REF-002 | [Upstream Kaggle dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) | Partially verified: the upstream listing was earlier observed to declare MIT; the direct response at the 2026-10-17 check supplied no readable content. This does not establish rights to every underlying image or the provenance of the downloaded revision. |
| REF-003 | [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) | Verified model metadata and processor configuration, with Apache-2.0 listed. This documentation work does not download weights or measure model quality. |
| REF-004 | [Official SmolVLM fine-tuning notebook](https://github.com/huggingface/smollm/blob/main/vision/finetuning/Smol_VLM_FT.ipynb) | Read adapter/quantization example. Original model and software assumptions differ from this project; code is a reference, not a validated drop-in recipe. |
| REF-005 | [TRL supervised fine-tuning trainer](https://huggingface.co/docs/trl/main/en/sft_trainer) | Verified vision dataset handling and image-token truncation guidance. Main documentation can differ from the pinned TRL release. |
| REF-006 | [uv dependency management](https://docs.astral.sh/uv/concepts/dependencies/) and [PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/) | Verified dependency groups, explicit indexes, and mutually exclusive runtime profile configuration. |
| REF-007 | [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/) and [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/main/en/installation) | Verified architecture/build guidance. No claim that every installed extension has been exercised. |
| REF-008 | [MIT license](https://opensource.org/license/mit) | Verified standard code license text. Does not govern third-party data/model artifacts automatically. |
| REF-009 | [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) | Verified driver/runtime compatibility guidance. Actual application support still needs runtime checks. |
| REF-010 | [NVIDIA GPU compute capability table](https://developer.nvidia.com/cuda/gpus) | Verified the RTX 5060 listing under compute capability 12.0; the architecture document owns the laptop-specific hardware observations. |
| REF-011 | [NVIDIA management-tool CUDA-version definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html#cuda-version) | Verified that the displayed version describes driver CUDA capability; it is not proof of the PyTorch wheel runtime or an installed toolkit. |
| REF-012 | [PyTorch 2.11.0 installation matrix](https://pytorch.org/get-started/previous-versions/#v2110) | Verified CUDA 12.8 and 13.0 wheel options with torchvision 0.26.0; runtime migration remains an issue #5 decision, not an executed upgrade. |

## Method references checked on 2026-10-18

| ID | Source | Verification and limits |
|---|---|---|
| REF-013 | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Read the original paper's abstract: frozen pretrained weights with trainable low-rank updates. Its published language-model results are rationale, not measured fashion-VLM improvements or laptop memory evidence. |
| REF-014 | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | Read the original abstract: adapter training through a frozen four-bit base, with NF4 and double quantization discussed. Does not establish compatibility, speed, memory savings, or task quality for the selected VLM and pinned stack. |
| REF-015 | [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora) and [custom-model guidance](https://huggingface.co/docs/peft/main/en/developer_guides/custom_models) | Read adapter configuration and model-targeting guidance. Exact language-only matches, trainable counts, and frozen components still need inspection of the loaded model. Main-branch guidance is not a tested recipe for the draft PEFT pin. |

## Implementation guidance follow-up (2026-10-18)

Rechecked [REF-005: TRL SFT Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer).
The current main guide warns that truncation can remove image tokens; it recommends
avoiding truncation unless its safety is verified. It also documents generation-mask
template requirements and, for VLM assistant-only loss, a Transformers >=5.18.0
requirement. The local dependency draft still pins Transformers 5.15.0 and TRL 1.10.0.

This is a reason to inspect the exact pinned trainer, processor, template, and collator
before accepting answer-only masks, not proof that every older implementation fails
or authorization to upgrade dependencies. The [feasibility plan](experiments.md#model-feasibility-and-execution-order)
requires inspection of actual masks, bounded image/text inputs, and safe image tokens.
Issue #5 owns dependency changes; no package install or model execution was performed.

## Evidence ownership and methodological limits

The main/supporting questions and scoring definitions in [experiments.md](experiments.md)
are project protocol proposals, not experimental findings from the papers. The
[dataset policy](dataset.md) owns vocabulary, visibility, and split rules; DR-002
records the proposed rationale. These sources do not supply validated splits,
independent visibility labels, guaranteed unknown behavior, or measured class
dependence for the local data.

Source checks do not establish this model's pretraining exposure to public fashion
images. A duplicate-aware split controls identified within-source leakage; it does
not prove that evaluation images were absent from pretraining. No universal LoRA/QLoRA
winner, hardware fit, training success, or quality gain is claimed.

## Local artifact observations

`data/raw/README.md` was read as downloaded source metadata. The two Parquet file
names and byte sizes were inspected without loading records. Their combined size
agrees with that card's `download_size`; this does not establish matching hashes,
actual row counts, decodable images, or a pinned repository revision.
The canonical inventory and unresolved provenance are in [dataset.md](dataset.md).

No dataset-content validation, split generation, model run, or new runtime check is
claimed for issue #4. GitHub Actions pins were inspected during earlier local draft
work; reviewing/accepting the workflow and observing hosted CI belong to issue #7.
