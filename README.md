# MV-SWIN-T

To fully exploit multi-view insights, we present a novel transformer-based multi-view network, MV-Swin-T, built upon the Swin Transformer architecture for mammographic image classification. 
Our contributions include:

\begin{enumerate}
    \item Designing a novel multi-view network entirely based on the transformer architecture, capitalizing on the benefits of transformer operations for enhanced performance.
    \item A novel "Multi-headed Dynamic Attention Block (MDA)" with fixed and shifted window features to enable self and cross-view information fusion from both CC  and MLO views of the same breast.
    \item Addressing the challenge of effectively combining data from multiple views or modalities, especially when images may not align correctly.
    \item We present results using the publicly available CBIS-DDSM And VinDr-Mammo dataset.
\end{enumerate}
