# Vesuvius-Challenge---Ink-Detection

Details can be find on the official [website](https://scrollprize.org/).

The end goal of Vesuvius Challenge to resurrect an ancient library from the ashes of a volcano. In this competition you are tasked with detecting ink from 3D X-ray scans and reading the contents. Thousands of scrolls were part of a library located in a Roman villa in Herculaneum, a town next to Pompeii. This villa was buried by the Vesuvius eruption nearly 2000 years ago. Due to the heat of the volcano, the scrolls were carbonized, and are now impossible to open without breaking them. These scrolls were discovered a few hundred years ago and have been waiting to be read using modern techniques.

The scrolls cannot be physically opened as explained [here](https://www.youtube.com/watch?v=PpNq2cFotyY).
![image](https://github.com/JeanJulesBigeard/Vesuvius-Challenge---Ink-Detection/assets/48935007/6d48cad4-3098-47d8-8aeb-4cac2c01a890)

This Kaggle competition hosts the Ink Detection Progress Prize, which is about the sub-problem of detecting ink from 3d x-ray scans of fragments of papyrus which became detached from some of the excavated scrolls. This subcontest is run on Kaggle since it's a more traditional data science / machine learning problem of building a model that can be verified against known ground truth data.

[anim2b_4.webm](https://github.com/JeanJulesBigeard/Vesuvius-Challenge---Ink-Detection/assets/48935007/b7d0c5e5-22c6-45dc-8658-516bf219c4c4)

The ink used in the Herculaneum scrolls does not show up readily in X-ray scans. But we have found that machine learning models can detect it. Luckily, we have ground truth data. Since the discovery of the Herculaneum Papyri almost 300 years ago, people have tried opening them, often with disastrous results. Many scrolls were destroyed in this process, but ink can be seen on some broken-off fragments, especially under infrared light.

The dataset contains 3d x-ray scans of four such fragments at 4µm resolution, made using a particle accelerator, as well as infrared photographs of the surface of the fragments showing visible ink. These photographs have been aligned with the x-ray scans. We also provide hand-labeled binary masks indicating the presence of ink in the photographs.

![image](https://github.com/JeanJulesBigeard/Vesuvius-Challenge---Ink-Detection/assets/48935007/570c25dc-7351-42ff-a144-50ae53b0c119)
