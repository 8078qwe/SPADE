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
        Panoptic Scene Graph Generation (PSG) integrates instance segmentation with relation understanding to capture pixel-level structural relationships in complex scenes. Although recent approaches leveraging pre-trained vision-language models (VLMs) have significantly improved performance in the open-vocabulary setting, they commonly ignore the inherent limitations of VLMs in spatial relation reasoning, such as difficulty in distinguishing object relative positions,  which results in suboptimal relation prediction. Motivated by the denoising diffusion model's inversion process in preserving the spatial structure of input images, we propose SPADE (SPatial-Aware Denoising-nEtwork) framework---a novel approach for open-vocabulary PSG. SPADE consists of two key steps: (1) inversion-guided calibration for the UNet adaptation, and (2) spatial-aware context reasoning. In the first step, we calibrate a general pre-trained teacher diffusion model into a PSG-specific denoising network with cross-attention maps derived during inversion through a lightweight LoRA-based fine-tuning strategy. In the second step, we develop a spatial-aware relation graph transformer that captures both local and long-range contextual information, facilitating the generation of high-quality relation queries. Extensive experiments on benchmark PSG and Visual Genome datasets demonstrate that SPADE outperforms state-of-the-art methods in both closed- and open-set scenarios, particularly for spatial relationship prediction.
        </div>
    </div>
</div>

---


# Background
**Panoptic Scene Graph Generation** (**PSG**) is a powerful paradigm that unifies instance segmentation and relationship prediction to enable pixel-level, structured scene understanding. Unlike traditional Scene Graph Generation (SGG), which typically operates on detected bounding boxes and a closed vocabulary, PSG requires simultaneously segmenting all entities and modeling their interactions as subject-predicate-object triples.

In recent years, approaches based on Vision-Language Models (VLMs) have made significant progress in **Open-vocabulary** scenarios, and recent advances in Visual Language Models (VLMs) such as CLIP and BLIP have pushed the frontiers of Open-vocabulary comprehension, enabling models to recognize novel objects and relationships beyond fixed categories. . However, while these models perform well in semantic recognition, these approaches still have limitations in spatial relational reasoning, especially when it comes to accurately understanding relationships between distant objects pairs.

# Motivation
Despite their impressive performance on open-world classification tasks, VLMs inherently lack spatial reasoning abilities because they are typically trained on image-caption datasets with limited spatial annotations. This limitation significantly affects their performance in predicting spatial predicates, especially when objects are far apart or exhibit subtle geometric relations.

Our investigation reveals a critical gap: ​​VLM-based PSG models underperform in spatial relation prediction​​, especially for distantly located objects (Fig. 1). Key observations include:

- ​​​​Distance sensitivity​​: Recall@50 for spatial predicates drops by 6–10% when objects are >1/3 image width apart.

Inspired by the inversion process in denoising diffusion models — known for preserving fine-grained spatial structures in images — we asked: Can we integrate spatial knowledge into VLM-based frameworks without sacrificing their strong open-vocabulary capabilities?

Follow this insight, we adapt diffusion models to PSG via:
- ​​Inversion-guided calibration​​ of cross-attention maps to inject spatial priors.
- ​​Dual-context reasoning​​ combining long-range and local spatial cues.
> *We hypothesize this synergy can overcome VLM limitations without sacrificing open-vocabulary generalization.*


![databias](/assets/databias.svg){: style="width: 500px; height: 300px; display: block; margin: 0 auto; margin-top: 50px; margin-bottom: 50px;"}


# Methods
We propose **SPADE** (**SPatial-Aware Denoising-nEtwork**), a novel two-stage framework that explicitly enhances spatial reasoning while retaining open-vocabulary recognition capabilities.

**Inversion-Guided UNet Calibration**
In the first stage, we leverage a pre-trained diffusion model as a spatially-aware teacher.  Using the inversion process, we extract cross-attention maps that serve as explicit spatial priors, guiding the adaptation of the UNet denoising backbone.  To maintain the open-vocabulary recognition power, we adopt a lightweight fine-tuning strategy called LoRA (Low-Rank Adaptation), updating only a small set of parameters in cross-attention layers.  This ensures that the pre-trained knowledge is preserved while injecting strong spatial cues.

**Spatial-Aware Context Reasoning**
In the second stage, we introduce a Spatial-Aware Relation Graph Transformer (RGT) to model both local and long-range context among segmented instances.  By constructing a spatial-semantic graph where nodes represent instance masks and edges encode spatial and semantic affinities, the RGT iteratively refines object features through a combination of graph convolutions and self-attention mechanisms.  This dual-context reasoning enables the model to capture subtle relationships that are crucial for accurate predicate prediction.


![framework](/assets/framework.svg){: style="width: 1000px; height: auto; display: block; margin: 0 auto; margin-top: 100px; margin-bottom: 100px;"}





## Citation
```
@article{SPADE,
  title={SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation with Long- and Local-range Context Reasoning},
  author={Xin Hu, Ke Qin, Guiduo Duan, Ming Li, Yuanfang Li, Tao He},
  journal={iccv},
  year={2025}
}
```
