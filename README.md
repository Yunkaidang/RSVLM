




## Contents
- [Install](#install)
- [Data](#data)
- [Models](#model)
- [Train](#train)
- [Evaluation](#evaluation)
## Install
Refer to the following command for installation.
```bash
git clone git@github.com:opendatalab/MF-RSVLM.git
cd MF-RSVLM
conda create -n mf-rsvlm 
conda activate mf-rsvlm
pip install -r requirements.txt
```


## Models
MF-RSVLM consists of a visual encoder, a projector layer, and a large language model (LLM). The visual encoder uses a pretrained [CLIP-14-336px](https://huggingface.co/openai/clip-vit-large-patch14-336), the projector layer is composed of two MLP layers, and the LLM is based on the pretrained [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5). The model is trained in two stages, as shown in the diagram below.

![](docs/images/mf-rsvlm_train_stage.png)


We provide not only the weights after the SFT stage but also the Pretrained weights.

| Name | Description|
|---|---|
|[MF-RSVLM_sft](https://huggingface.co/FitzPC/mf-rsvlm_7B) | The LLM and MLP weights obtained from the SFT stage| 
|[MF-RSVLM_pretrain](https://huggingface.co/FitzPC/mf-rsvlm_7b_pretrain_mlp_llm/tree/main) | The LLM and MLP weights obtained from the Pretraining stage.|
|[CLIP_pretrain](https://huggingface.co/FitzPC/mf-rsvlm_7b_pretrain_vit)|The CLIP weights obtained from the  Pretraining stage.|


## Train
MF-RSVLM model training consists of two stages: (1) Pretrain stage: use our VersaD dataset with 1.4M image-text pairs to finetune the vision encoder, projector, and the LLM to align the textual and visual modalities; (2) Supervised Fine-Tuning（SFT） stage: finetune the projector and LLM to teach the model to follow multimodal instructions. 
### Pretrain
First, you should download the [MLP projector](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/tree/main) pretrained by LLaVA-1.5. Because a rough modality alignment process is beneficial before using high quality detailed captions for modality alignment.

You can run `sh scripts/rs/slurm_pretrain.sh` to pretrain the model. Remember to specify the projector path in the script. In this stage, we fine-tuned the second half of the vision encoder's blocks, projector, and LLM.

In our setup we used 16 A100 (80G) GPUs and the whole pre-training process lasted about 10 hours. You can adjust the number of gradient accumulation steps to reduce the number of GPUs.

In the `sh scripts/rs/slurm_pretrain.sh`, you need to revise three paths:
```bash
DATA_DIR=pretrain_base # directory of VersaD dataset
export LIST_FILE=${DATA_DIR}/list_pretrain.json # json file of VersaD data  
export CKPT_PATH=weight_path # llava-1.5 MLP weight path
export SAVE_PATH=mf-rsvlm-7b_pretrained # file save path
```
### Supervised Fine-Tuning
In this stage, we finetune the projector and LLM with our [MF-RSVLM_SFT](https://huggingface.co/datasets/FitzPC/MF-RSVLM_dataset_sft) dataset. 

In our setup we used 8 A100 (80G) GPUs and the whole sft process lasted about 4 hours. You can adjust the number of gradient accumulation steps to reduce the number of GPUs.

You can run `sh scripts/rs/slurm_finetune.sh` to finetune the model, and you need to revise three paths:
```bash
DATA_DIR=sft_base # directory of mf-rsvlm-sft dataset
export LIST_FILE=${DATA_DIR}/list_sft.json # json file of sft data  
CKPT=mf-rsvlm-7b_pretrained # pretrain weight path
export SAVE_PATH=mf-rsvlm-7b_sft # file save path
```

## Evaluation 
In order to facilitate the use of remote sensing vision-language large models, we have developed a specialized evaluation project [RSEvalKit](https://github.com/fitzpchao/RSEvalKit) for remote sensing large models. Please refer to the following command for installation.

```sh
git clone https://github.com/fitzpchao/RSEvalKit
cd RSEvalKit
conda create -n rseval
conda activate rseval
pip install -r requirements.txt
```
All evaluation tasks for this paper are implemented in RSEval  and can be evaluated with one click. First, you need to download our [model weights](#models) and [MF-RSVLM_Eval data](docs/Data.md#MF-RSVLM_Eval-Dataset ), then follow the [instructions](https://github.com/fitzpchao/RSEvalKit/blob/master/README.md) to complete the evaluation.

## Citation
```bibtex
@article{dang2025fuse,
  title={FUSE-RSVLM: Feature Fusion Vision-Language Model for Remote Sensing},
  author={Dang, Yunkai and Wang, Donghao and Yang, Jiacheng and Jiang, Yifan and Zhu, Meiyi and Yang, Yuekun and Wang, Cong and Fan, Qi and Li, Wenbin and Gao, Yang},
  journal={arXiv preprint arXiv:2512.24022},
  year={2025}
}
```

## Acknowledgement
We gratefully acknowledge these wonderful works：
- [Vicuna](https://github.com/lm-sys/FastChat#vicuna-weights)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V)
- [LLaMA](https://github.com/facebookresearch/llama)

## License

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and Gemini. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
