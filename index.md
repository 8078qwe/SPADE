---
layout: project_page
permalink: /
title: "SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation with Long- and Local-range Context Reasoning"
authors:
  - name: Xin Hu
    link: https://openreview.net/profile?id=~XIN_Hu8
    superscript: 1
  - name: Ke Qin
    link: https://www.scse.uestc.edu.cn/info/1081/11199.htm
    superscript: 1,2
  - name: Guiduo Duan
    link: https://www.scse.uestc.edu.cn/info/1081/11210.htm
    superscript: 1,2
  - name: Ming Li
    link: https://ming1993li.github.io/
    superscript: 4
  - name: Yuan-Fang Li
    link: https://liyuanfang.github.io/
    superscript: 3
  - name: Tao He
    link: https://ht014.github.io/
    superscript: 1,2,*

affiliations: |
  <sup>1</sup> The Laboratory of Intelligent Collaborative Computing of UESTC<br>
  <sup>2</sup> Ubiquitous Intelligence and Trusted Services Key Laboratory of Sichuan Province<br>
  <sup>3</sup> Faculty of Information Technology, Monash University<br>
  <sup>4</sup> Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)
  <br><sup>*</sup> Corresponding author

accepted:
    The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025
paper: https://openaccess.thecvf.com/content/CVPR2025/html/Yin_Knowledge-Aligned_Counterfactual-Enhancement_Diffusion_Perception_for_Unsupervised_Cross-Domain_Visual_Emotion_Recognition_CVPR_2025_paper.html
# video: https://www.youtube.com/results?search_query=turing+machine
code: https://github.com/yinwen2019/KCDP
# data: https://huggingface.co/docs/datasets
slide: https://yinwen2019.github.io/ucdver/assets/slide.pptx
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h1>Abstract</h1>
        <div class="content has-text-justified">
        Panoptic Scene Graph Generation (PSG) integrates instance segmentation with relation understanding to capture pixel-level structural relationships in complex scenes. Although recent approaches leveraging pre-trained vision-language models (VLMs) have significantly improved performance in the open-vocabulary setting, they commonly ignore the inherent limitations of VLMs in spatial relation reasoning, such as difficulty in distinguishing object relative positions,  which results in suboptimal relation prediction. Motivated by the denoising diffusion model's inversion process in preserving the spatial structure of input images, we propose \textbf{SPADE} (\underline{SP}atial-\underline{A}ware \underline{D}enoising-n\underline{E}twork) framework---a novel approach for open-vocabulary PSG. SPADE consists of two key steps: (1) inversion-guided calibration for the UNet adaptation, and (2) spatial-aware context reasoning. In the first step, we calibrate a general pre-trained teacher diffusion model into a PSG-specific denoising network with cross-attention maps derived during inversion through a lightweight LoRA-based fine-tuning strategy. In the second step, we develop a spatial-aware relation graph transformer that captures both local and long-range contextual information, facilitating the generation of high-quality relation queries. Extensive experiments on benchmark PSG and Visual Genome datasets demonstrate that SPADE outperforms state-of-the-art methods in both closed- and open-set scenarios, particularly  for spatial relationship prediction.
        </div>
    </div>
</div>

---


# Background
**Visual Emotion Recognition** (**VER**), a fundamental task in artificial intelligence and human-computer interaction, aims to identify human emotions through visual cues, such as facial expressions, body language  and contextual scene features. Existing VER methods typically focus on realistic images and have gained considerable advancements on a suite of datasets such as **EmoSet** and **Emotion6**. Unfortunately, current VER models cannot handle emotion recognition in these new domains due to the significant emotional expression variability between domains and an affective distribution shift. In this paper, we introduce a new challenging task **Unsupervised Cross-Domain Visual Emotion Recognition** (**UCDVER**), where a model is trained with labeled source-domain data (e.g., realistic images) and unlabeled target domain data (e.g., stickers), but is employed to recognize emotion in the target domain.

# Motivation
Taking the stickers and realistic images as an example shown in Figure 1, two key challenges arise:
- **Emotional expression variability**: Emotional expressions vary greatly. Realistic images reflect emotions expressed by real humans, while stickers exaggerate or simplify emotions, often focusing on single or multiple virtual elements.
- **Affective distribution shift**: According to the _Emotional Causality theory_, an emotion is embedded in a sequence involving (i) _external event_; (ii) _emotional state_; and (iii) _physiological response_. Stickers or emojis emphasize the last two, i.e. (ii) and (iii) , while the emotion in realistic images is often linked to the external context surrounding the subject(s).


![databias](/assets/databias.svg){: style="width: 500px; height: 300px; display: block; margin: 0 auto; margin-top: 50px; margin-bottom: 50px;"}


# Methods
We propose a **Knowledge-aligned Counterfactual-enhancement Diffusion Perception** (**KCDP**) framework, which projects affective cues into a domain-agnostic knowledge space and performs domain-adaptive visual affective perception by a diffusion model.  

Briefly, KCDP is composed of two primary modules: **Knowledge-Alignment Diffusion
Affective Perception** (**KADAP**) and **Counterfactual-Enhanced Language-Image Emotional Alignment** (**CLIEA**). The \textbf{KADAP} module focuses on learning domain-agnostic knowledge and making robust predictions based on an MoE predictor , while the **CLIEA** module generates high-quality pseudo-labels for effective training .


![framework](/assets/framework.svg){: style="width: 1000px; height: auto; display: block; margin: 0 auto; margin-top: 100px; margin-bottom: 100px;"}


The CLIEA strategy is designed to generate high-quality emotional pseudo-labels for  the target domain. CLIEA is inspired by the causal relationships underlying language-image emotional alignment.


![framework](/assets/casusalgraph.svg){: style="width: 500px; height: auto; display: block; margin: 0 auto; margin-top: 50px; margin-bottom: 50px;"}


## Citation
```
@article{ucdver,
  title={Knowledge-Aligned Counterfactual-Enhancement Diffusion Perception for Unsupervised Cross-Domain Visual Emotion Recognition},
  author={Wen Yin, Yong Wang, Guiduo Duan, Dongyang Zhang, Xin Hu, Yuan-Fang Li, Tao He},
  journal={CVPR},
  year={2025}
}
```
