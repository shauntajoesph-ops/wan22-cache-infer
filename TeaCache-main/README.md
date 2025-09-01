# [CVPR 2025 Highlight] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model

<div class="is-size-5 publication-authors", align="center",>
            <span class="author-block">
              <a href="https://liewfeng.github.io" target="_blank">Feng Liu</a><sup>1</sup><sup>*</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com.hk/citations?user=ZO3OQ-8AAAAJ" target="_blank">Shiwei Zhang</a><sup>2</sup><sup>†</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://jeffwang987.github.io" target="_blank">Xiaofeng Wang</a><sup>1,3</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://weilllllls.github.io" target="_blank">Yujie Wei</a><sup>4</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="http://haonanqiu.com" target="_blank">Haonan Qiu</a><sup>5</sup>
            </span>
            <br>
            <span class="author-block">
              <a href="https://callsys.github.io/zhaoyuzhong.github.io-main" target="_blank">Yuzhong Zhao</a><sup>1</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com.sg/citations?user=16RDSEUAAAAJ" target="_blank">Yingya Zhang</a><sup>2</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=tjEfgsEAAAAJ&hl=en&oi=ao" target="_blank">Qixiang Ye</a><sup>1</sup>,&nbsp;
            </span>
            <span class="author-block">
              <a href="https://scholar.google.com/citations?user=0IKavloAAAAJ&hl=en&oi=ao" target="_blank">Fang Wan</a><sup>1</sup><sup>‡</sup>
            </span>
          </div>

<div class="is-size-5 publication-authors", align="center">
            <span class="author-block"><sup>1</sup>University of Chinese Academy of Sciences,&nbsp;</span>
            <span class="author-block"><sup>2</sup>Alibaba Group</span>
            <br>
            <span class="author-block"><sup>3</sup>Institute of Automation, Chinese Academy of Sciences</span>
            <br>
            <span class="author-block"><sup>4</sup>Fudan University,&nbsp;</span>
            <span class="author-block"><sup>5</sup>Nanyang Technological University</span>
          </div>


<div class="is-size-5 publication-authors", align="center">
            (* Work was done during internship at Alibaba Group. † Project Leader. ‡ CorresCorresponding author.)
          </div>

<h5 align="center">

[![hf_paper](https://img.shields.io/badge/🤗-Paper%20In%20HF-red.svg)](https://huggingface.co/papers/2411.19108)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.19108-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.19108) 
[![Home Page](https://img.shields.io/badge/Project-<Website>-blue.svg)](https://liewfeng.github.io/TeaCache/) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE) 
[![github](https://img.shields.io/github/stars/LiewFeng/TeaCache.svg?style=social)](https://github.com/LiewFeng/TeaCache/)

</h5>


![visualization](./assets/tisser.png)

## 🫖 Introduction 
We introduce Timestep Embedding Aware Cache (TeaCache), a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps, thereby accelerating the inference. TeaCache works well for Video Diffusion Models, Image Diffusion models and Audio Diffusion Models. For more details and results, please visit our [project page](https://liewfeng.github.io/TeaCache/).

## 🔥 Latest News 
- **If you like our project, please give us a star ⭐ on GitHub for the latest update.**
- [2025/06/08] 🔥 Update coefficients of [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0). Thanks [@spawner1145](https://github.com/spawner1145).
- [2025/05/26] 🔥 Support [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0). Thanks [@spawner1145](https://github.com/spawner1145). 
- [2025/05/25] 🔥 Support [HiDream-I1](https://github.com/HiDream-ai/HiDream-I1). Thanks [@YunjieYu](https://github.com/YunjieYu). 
- [2025/04/14] 🔥 Update coefficients of [CogVideoX1.5](https://github.com/THUDM/CogVideo). Thanks [@zishen-ucap](https://github.com/zishen-ucap).
- [2025/04/05] 🎉 Recommended as a **highlight** in CVPR 2025, top 16.8% in accepted papers and top 3.7% in all papers.
- [2025/03/13] 🔥 Optimized TeaCache for [Wan2.1](https://github.com/Wan-Video/Wan2.1). Thanks [@zishen-ucap](https://github.com/zishen-ucap).
- [2025/03/05] 🔥 Support [Wan2.1](https://github.com/Wan-Video/Wan2.1) for both T2V and I2V.
- [2025/02/27] 🎉 Accepted in **CVPR 2025**.
- [2025/01/24] 🔥 Support [Cosmos](https://github.com/NVIDIA/Cosmos) for both T2V and I2V. Thanks [@zishen-ucap](https://github.com/zishen-ucap). 
- [2025/01/20] 🔥 Support [CogVideoX1.5-5B](https://github.com/THUDM/CogVideo) for both T2V and I2V. Thanks [@zishen-ucap](https://github.com/zishen-ucap). 
- [2025/01/07] 🔥 Support [TangoFlux](https://github.com/declare-lab/TangoFlux). TeaCache works well for Audio Diffusion Models!
- [2024/12/30] 🔥 Support [Mochi](https://github.com/genmoai/mochi) and [LTX-Video](https://github.com/Lightricks/LTX-Video) for Video Diffusion Models. Support [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X) for Image Diffusion Models.
- [2024/12/27] 🔥 Support [FLUX](https://github.com/black-forest-labs/flux). TeaCache works well for Image Diffusion Models!
- [2024/12/26] 🔥 Support [ConsisID](https://github.com/PKU-YuanGroup/ConsisID). Thanks [@SHYuanBest](https://github.com/SHYuanBest). 
- [2024/12/24] 🔥 Support [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).
- [2024/12/19] 🔥 Support [CogVideoX](https://github.com/THUDM/CogVideo).
- [2024/12/06] 🎉 Release the [code](https://github.com/LiewFeng/TeaCache) of TeaCache. Support [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) and [Latte](https://github.com/Vchitect/Latte).
- [2024/11/28] 🎉 Release the [paper](https://arxiv.org/abs/2411.19108) of TeaCache.

## 🧩 Community Contributions  
If you develop/use TeaCache in your projects and you would like more people to see it, please inform us.(liufeng20@mails.ucas.ac.cn)

**Model**
- [FramePack](https://github.com/lllyasviel/FramePack) supports TeaCache. Thanks [@lllyasviel](https://github.com/lllyasviel).
- [FastVideo](https://github.com/hao-ai-lab/FastVideo) supports TeaCache. Thanks [@BrianChen1129](https://github.com/BrianChen1129) and [@jzhang38](https://github.com/jzhang38).
- [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) supports TeaCache. Thanks [@hkunzhe](https://github.com/hkunzhe) and [@bubbliiiing](https://github.com/bubbliiiing).
- [Ruyi-Models](https://github.com/IamCreateAI/Ruyi-Models) supports TeaCache. Thanks [@cellzero](https://github.com/cellzero).
- [ConsisID](https://github.com/PKU-YuanGroup/ConsisID) supports TeaCache. Thanks [@SHYuanBest](https://github.com/SHYuanBest).

**ComfyUI**
- [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) for TeaCache. Thanks [@YunjieYu](https://github.com/YunjieYu).
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) supports TeaCache4Wan2.1. Thanks [@kijai](https://github.com/kijai).
- [ComfyUI-TangoFlux](https://github.com/LucipherDev/ComfyUI-TangoFlux) supports TeaCache. Thanks [@LucipherDev](https://github.com/LucipherDev).
- [ComfyUI_Patches_ll](https://github.com/lldacing/ComfyUI_Patches_ll) supports TeaCache. Thanks [@lldacing](https://github.com/lldacing).
- [Comfyui_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset) supports TeaCache. Thanks [@TTPlanetPig](https://github.com/TTPlanetPig).
- [ComfyUI-TeaCacheHunyuanVideo](https://github.com/facok/ComfyUI-TeaCacheHunyuanVideo) for TeaCache4HunyuanVideo. Thanks [@facok](https://github.com/facok).
- [ComfyUI-HunyuanVideoWrapper](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) supports TeaCache4HunyuanVideo. Thanks [@kijai](https://github.com/kijai), [ctf05](https://github.com/ctf05) and [DarioFT](https://github.com/DarioFT).


**Parallelism**
- [Teacache-xDiT](https://github.com/MingXiangL/Teacache-xDiT) for multi-gpu inference. Thanks [@MingXiangL](https://github.com/MingXiangL).

**Engine**
- [SD.Next](https://github.com/vladmandic/sdnext) supports TeaCache. Thanks [@vladmandic](https://github.com/vladmandic).
- [DiffSynth Studio](https://github.com/modelscope/DiffSynth-Studio) supports TeaCache. Thanks [@Artiprocher](https://github.com/Artiprocher).
  
## 🎉 Supported Models 
**Text to Video**
- [TeaCache4Wan2.1](./TeaCache4Wan2.1/README.md)
- [TeaCache4Cosmos](./eval/TeaCache4Cosmos/README.md)
- [TeaCache4CogVideoX1.5](./TeaCache4CogVideoX1.5/README.md)
- [TeaCache4LTX-Video](./TeaCache4LTX-Video/README.md)
- [TeaCache4Mochi](./TeaCache4Mochi/README.md)
- [TeaCache4HunyuanVideo](./TeaCache4HunyuanVideo/README.md)
- [TeaCache4CogVideoX](./eval/teacache/README.md)
- [TeaCache4Open-Sora](./eval/teacache/README.md)
- [TeaCache4Open-Sora-Plan](./eval/teacache/README.md)
- [TeaCache4Latte](./eval/teacache/README.md)

 **Image to Video** 
- [TeaCache4Wan2.1](./TeaCache4Wan2.1/README.md)
- [TeaCache4Cosmos](./eval/TeaCache4Cosmos/README.md)
- [TeaCache4CogVideoX1.5](./TeaCache4CogVideoX1.5/README.md)
- [TeaCache4ConsisID](./TeaCache4ConsisID/README.md)

 **Text to Image**
- [TeaCache4Lumina2](./TeaCache4Lumina2/README.md)
- [TeaCache4HiDream-I1](./TeaCache4HiDream-I1/README.md)
- [TeaCache4FLUX](./TeaCache4FLUX/README.md)
- [TeaCache4Lumina-T2X](./TeaCache4Lumina-T2X/README.md)

 **Text to Audio**
- [TeaCache4TangoFlux](./TeaCache4TangoFlux/README.md)

## 🤖 Instructions for Supporting Other Models 
- **Welcome for PRs to support other models.**
- If the custom model is based on or has similar model structure to the models we've supported, you can try to directly transfer TeaCache to the custom model. For example,  rescaling coefficients for CogVideoX-5B can be directly applied to CogVideoX1.5, ConsisID and rescaling coefficients for FLUX can be directly applied to TangoFlux.
- Otherwise, you can refer to these successful attempts, e.g., [1](https://github.com/ali-vilab/TeaCache/issues/20), [2](https://github.com/ali-vilab/TeaCache/issues/18).

## 💐 Acknowledgement 

This repository is built based on [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Diffusers](https://github.com/huggingface/diffusers), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Latte](https://github.com/Vchitect/Latte), [CogVideoX](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), [FLUX](https://github.com/black-forest-labs/flux), [Mochi](https://github.com/genmoai/mochi), [LTX-Video](https://github.com/Lightricks/LTX-Video), [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X), [TangoFlux](https://github.com/declare-lab/TangoFlux), [Cosmos](https://github.com/NVIDIA/Cosmos), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HiDream-I1](https://github.com/HiDream-ai/HiDream-I1) and [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0). Thanks for their contributions!

## 🔒 License 

* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
* For [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), [Diffusers](https://github.com/huggingface/diffusers), [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), [Latte](https://github.com/Vchitect/Latte), [CogVideoX](https://github.com/THUDM/CogVideo), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), [FLUX](https://github.com/black-forest-labs/flux), [Mochi](https://github.com/genmoai/mochi), [LTX-Video](https://github.com/Lightricks/LTX-Video), [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X), [TangoFlux](https://github.com/declare-lab/TangoFlux), [Cosmos](https://github.com/NVIDIA/Cosmos), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HiDream-I1](https://github.com/HiDream-ai/HiDream-I1) and [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0), please follow their LICENSE.
* The service is a research preview. Please contact us if you find any potential violations. (liufeng20@mails.ucas.ac.cn)

## 📖 Citation 
If you find TeaCache is useful in your research or applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry.

```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```


