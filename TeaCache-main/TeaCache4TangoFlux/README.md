<!-- ## **TeaCache4TangoFlux** -->
# TeaCache4TangoFlux

[TeaCache](https://github.com/LiewFeng/TeaCache) can speedup [TangoFlux](https://github.com/declare-lab/TangoFlux) 2x without much audio quality degradation, in a training-free manner.

## 📈 Inference Latency Comparisons on a Single A800


|      TangoFlux      |        TeaCache (0.25)       |    TeaCache (0.4)    |
|:-------------------:|:----------------------------:|:--------------------:|
|      ~4.08 s        |        ~2.42 s                |     ~1.95 s         |

## Installation

```shell
pip install git+https://github.com/declare-lab/TangoFlux
```

## Usage

You can modify the thresh in line 266 to obtain your desired trade-off between latency and audio quality. For single-gpu inference, you can use the following command:

```bash
python teacache_tango_flux.py
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

We would like to thank the contributors to the [TangoFlux](https://github.com/declare-lab/TangoFlux).