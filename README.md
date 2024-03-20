# A Multimodal Framework to Detect Target Aware Aggression in Memes

Internet memes have gained immense traction as a medium for individuals to convey emotions, thoughts, and perspectives on social media. While memes often serve as sources of humor and entertainment, they can also propagate offensive, incendiary, or harmful content, deliberately targeting specific individuals or communities. Identifying such memes is challenging because of their satirical and cryptic characteristics. Most contemporary research on memes’ detrimental facets is skewed towards high-resource languages, often sidelining the unique challenges tied to low-resource languages, such as Bengali. To facilitate this research in low-resource languages, this paper presents a novel dataset **MIMOSA** (MultIMOdal aggreSsion dAtaset) in Bengali. **MIMOSA** encompasses 4,848 annotated memes across five aggression target categories: Political, Gender, Religious, Others, and non-aggressive. We also propose **MAF** (Multimodal Attentive Fusion), a simple yet effective approach that uses multimodal context to detect the aggression targets. MAF captures the selective modality-specific features of the input meme and jointly evaluates them with individual modality features. Experiments on **MIMOSA** exhibit that the proposed method outperforms several state-of-the-art rivaling approaches. 

**Author:** Shawly Ahsan, Eftekhar Hossain, Omar Sharif, Avishek Das, Mohammed Moshiul Hoque, M. Ali Akber Dewan 

**Venue:** 18th Conference of the European Chapter of the Association for Computational Linguistics

[Paper Link](https://aclanthology.org/2024.eacl-long.153/)

## Citation

If you are using our code, dataset or models for your research work then please cite our paper below:
```
@inproceedings{ahsan-etal-2024-multimodal,
    title = "A Multimodal Framework to Detect Target Aware Aggression in Memes",
    author = "Ahsan, Shawly  and
      Hossain, Eftekhar  and
      Sharif, Omar  and
      Das, Avishek  and
      Hoque, Mohammed Moshiul  and
      Dewan, M.",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.153",
    pages = "2487--2500",
    abstract = "Internet memes have gained immense traction as a medium for individuals to convey emotions, thoughts, and perspectives on social media. While memes often serve as sources of humor and entertainment, they can also propagate offensive, incendiary, or harmful content, deliberately targeting specific individuals or communities. Identifying such memes is challenging because of their satirical and cryptic characteristics. Most contemporary research on memes{'} detrimental facets is skewed towards high-resource languages, often sidelining the unique challenges tied to low-resource languages, such as Bengali. To facilitate this research in low-resource languages, this paper presents a novel dataset MIMOSA (MultIMOdal aggreSsion dAtaset) in Bengali. MIMOSA encompasses 4,848 annotated memes across five aggression target categories: Political, Gender, Religious, Others, and non-aggressive. We also propose MAF (Multimodal Attentive Fusion), a simple yet effective approach that uses multimodal context to detect the aggression targets. MAF captures the selective modality-specific features of the input meme and jointly evaluates them with individual modality features. Experiments on MIMOSA exhibit that the proposed method outperforms several state-of-the-art rivaling approaches. Our code and data are available at https://github.com/shawlyahsan/Bengali-Aggression-Memes.",
}
```
