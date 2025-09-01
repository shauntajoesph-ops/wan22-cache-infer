<!-- ## **TeaCache4LuminaT2X** -->
# TeaCache4Lumina2

[TeaCache](https://github.com/LiewFeng/TeaCache) can speedup [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) without much visual quality degradation, in a training-free manner. The following image shows the experimental results of Lumina-Image-2.0 and TeaCache with different versions: v1(0 (original), 0.2 (1.25x speedup), 0.3 (1.5625x speedup), 0.4 (2.0833x speedup), 0.5 (2.5x speedup).) and v2(Lumina-Image-2.0 (~25 s), TeaCache (0.2) (~16.7 s, 1.5x speedup), TeaCache (0.3) (~15.6 s, 1.6x speedup), TeaCache (0.5) (~13.79 s, 1.8x speedup), TeaCache (1.1) (~11.9 s, 2.1x speedup)).

The v1 coefficients 
`[393.76566581,−603.50993606,209.10239044,−23.00726601,0.86377344]`
 exhibit poor quality at low L1 values but perform better with higher L1 settings, though at a slower speed. The v2 coefficients 
`[225.7042019806413,−608.8453716535591,304.1869942338369,124.21267720116742,−1.4089066892956552]`
, however, offer faster computation and better quality at low L1 levels but incur significant feature loss at high L1 values.

You can change the value in line 72 to switch versions

## v1
<p align="center">
    <img src="https://github.com/user-attachments/assets/d2c87b99-e4ac-4407-809a-caf9750f41ef" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/411ff763-9c31-438d-8a9b-3ec5c88f6c27" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/e57dfb60-a07f-4e17-837e-e46a69d8b9c0" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/6e3184fe-e31a-452c-a447-48d4b74fcc10" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/d6a52c4c-bd22-45c0-9f40-00a2daa85fc8" width="150" style="margin: 5px;">
</p>

## v2
<p align="center">
    <img src="https://github.com/user-attachments/assets/aea9907b-830e-497b-b968-aaeef463c7ef" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/0e258295-eaaa-49ce-b16f-bba7f7ada6c1" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/44600f22-3fd4-4bc4-ab00-29b0ed023d6d" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/bcb926ab-95fd-4c83-8b46-f72581a3359e" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/ec8db28e-0f9b-4d56-9096-fdc8b3c20f4b" width="150" style="margin: 5px;">
</p>

## 📈 Inference Latency Comparisons on a single 4090 (step 50)
## v1
|      Lumina-Image-2.0      |        TeaCache (0.2)       |    TeaCache (0.3)    |     TeaCache (0.4)    |     TeaCache (0.5)    |
|:-------------------------:|:---------------------------:|:--------------------:|:---------------------:|:---------------------:|
|         ~25 s             |        ~20 s                |     ~16 s            |       ~12 s             |       ~10 s             |

## v2
|      Lumina-Image-2.0      |        TeaCache (0.2)       |    TeaCache (0.3)    |     TeaCache (0.5)    |     TeaCache (1.1)    |
|:-------------------------:|:---------------------------:|:--------------------:|:---------------------:|:---------------------:|
|         ~25 s             |        ~16.7 s                |     ~15.6 s            |       ~13.79 s             |       ~11.9 s             |

## Installation

```shell
pip install --upgrade diffusers[torch] transformers protobuf tokenizers sentencepiece
pip install flash-attn --no-build-isolation
```

## Usage

You can modify the thresh in line 154 to obtain your desired trade-off between latency and visul quality. For single-gpu inference, you can use the following command:

```bash
python teacache_lumina2.py
```

## Citation
If you find TeaCache is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```

## Acknowledgements

We would like to thank the contributors to the [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) and [Diffusers](https://github.com/huggingface/diffusers).
