# MV-Swin-T

In this article, we present a novel transformer-based multi-view network, MV-Swin-T, built upon the Swin Transformer architecture for mammographic image classification to fully exploit multi-view insights.
Our contributions include:

- Designing a novel multi-view network entirely based on the transformer architecture, capitalizing on the benefits of transformer operations for enhanced performance.
- A novel "Multi-headed Dynamic Attention Block (MDA)" with fixed and shifted window features to enable self and cross-view information fusion from both CC  and MLO views of the same breast.
- Addressing the challenge of effectively combining data from multiple views or modalities, especially when images may not align correctly.
- We present results using the publicly available CBIS-DDSM And VinDr-Mammo dataset.

### Results
![image](https://github.com/prithuls/MV-SWIN-T/assets/43958517/e60a2270-1203-48ac-8d67-430cc2ed6368)

![image](https://github.com/prithuls/MV-SWIN-T/assets/43958517/408f6de0-5dca-43ba-b129-8031b70a81e5)

### Figures

![cross_attention](https://github.com/prithuls/MV-Swin-T/assets/43958517/af72ac57-7c20-4584-85cc-9728fdd9234e)



![window_attention](https://github.com/prithuls/MV-Swin-T/assets/105523359/fa856cd4-ee0e-4e6e-9e7a-af8c2dd222bf)


title: FSquaDRA
abstract: Fast detection of repackaged Android applications based on the comparison of resource files included into the package.
authors:
  - family-names: Zhauniarovich
    given-names: Yury
    orcid: "https://orcid.org/0000-0001-9116-0728"
cff-version: 1.2.0
date-released: "2013-11-30"
identifiers:
  - type: url
    value: "https://github.com/zyrikby/FSquaDRA/tree/dc42c93991240da0fc9f1081e72be3eeb17d2638"
    description: Latest version
keywords:
  - research
  - "detection repackaged Android applications"
  - "resouce files"
license: Apache-2.0
message: If you use this software, please cite it using these metadata.
repository-code: "https://github.com/zyrikby/FSquaDRA"
preferred-citation:
  title: "FSquaDRA: Fast Detection of Repackaged Applications"
  type: conference-paper
  authors:
  - family-names: "Zhauniarovich"
    given-names: "Yury"
  - family-names: "Gadyatskaya"
    given-names: "Olga"
  - family-names: "Crispo"
    given-names: "Bruno"
  - family-names: "La Spina"
    given-names: "Francesco"
  - family-names: "Moser"
    given-names: "Ermanno"
  collection-title: "28th Annual IFIP WG 11.3 Working Conference on Data and Applications Security and Privacy" # booktitle
  collection-type: "proceedings"
  conference:
    name: "DBSec" # series
  doi: "10.1007/978-3-662-43936-4_9"
  start: 131 # First page number
  end: 146 # Last page number
  year: 2014
