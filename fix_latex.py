import re

with open('report_architecture.tex', 'r') as f:
    content = f.read()

new_text = r"""\subsubsection{Sparse Diffusion Targeting}

A key observation in our early experiments is that directly applying diffusion enhancement to the entire hard mask leads to overuse of generative modification, which damages surrounding background consistency and introduces seam artifacts.
Therefore, we explicitly restrict diffusion to a sparse subset of difficult pixels instead of the whole candidate region.

Our pipeline first applies ProPainter as the main restoration module and then constructs a \texttt{diffusion\_target} mask for local enhancement.
The target mask is obtained from candidate regions after excluding borrowable areas, followed by confidence gating, morphological filtering, and connected-component cleanup.
When the target region becomes too large, we enforce an upper bound on the ratio between diffusion-target pixels and hard-mask pixels, so that only the most difficult pixels are retained for diffusion refinement.
The difficulty score combines low support fraction and high color inconsistency, thereby prioritizing pixels that are less reliably restored by propagation-based methods.

This sparse design is supported by quantitative evidence.
Here, \texttt{target/hard} denotes the ratio between the number of pixels selected for diffusion refinement and the number of pixels in the hard mask.
We report both the mean ratio over all evaluated samples and the 90th percentile (p90), which reflects the upper-end behavior in high-coverage cases.
Before introducing the ratio constraint, the average \texttt{target/hard} ratio was approximately \(0.915\), indicating that diffusion was effectively applied to almost the entire hard region.
After the sparse strategy was introduced, this ratio was reduced to \(0.35\) or \(0.25\), confirming that diffusion was restricted to a much smaller and more selective subset of pixels.

\begin{table}[h]
\centering
\caption{Effect of sparse targeting on diffusion coverage. Lower values indicate more selective diffusion refinement.}
\label{tab:sparse_target}
\begin{tabular}{lcc}
\toprule
Setting & target/hard mean & target/hard p90 \\
\midrule
Before sparsification & 0.915 & 0.994 \\
Sparse target (\(0.35\)) & 0.350 & 0.350 \\
Sparse target (\(0.25\)) & 0.250 & 0.250 \\
\bottomrule
\end{tabular}
\end{table}

Among these settings, the sparse-\(0.35\) configuration was considered the most balanced in terms of quality and stability, and was therefore kept as the default setting inside the diffusion-enhanced branch.

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{results/part3/wildvideo/blending_comparison_labeled_horizontal.png}
    \caption{Sparse diffusion targeting. From left to right: hard mask, diffusion target, and enhanced result.}
    \label{fig:sparse_target}
\end{figure}

\subsubsection{Improved Boundary Blending}

Besides overuse of diffusion, another major artifact in our early results is the visible halo or bright boundary around refined regions.
Our analysis suggests that this issue comes from the interaction between VAE decoding artifacts and an overly rigid alpha blending scheme, which effectively pastes boundary noise back into the restored frame.

To alleviate this problem, we adopted a distance-transform-based alpha blending strategy.
Instead of using a hard or poorly shaped transition band, the new design assigns low alpha values near the outer boundary and gradually increases the replacement weight toward the interior of the refined region.
This creates a much smoother geometric transition between the enhanced patch and the original restored frame.

In practice, this modification substantially reduces halo artifacts and improves single-frame visual naturalness. It plummeted the outside change L1 mean from \(0.0091\) to \(0.0023\) (a 74\% reduction) and the seam change L1 mean from \(0.6508\) to \(0.2841\).
It is one of the clearest technical improvements obtained in Part 3.
Nevertheless, this improvement mainly benefits spatial quality on individual frames, and does not by itself resolve the temporal instability introduced by frame-wise diffusion enhancement.

\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{figures/part3_boundary_blending.png}
    \caption{Improved boundary blending. The distance-transform alpha reduces visible halo and seam artifacts.}
    \label{fig:boundary_blending}
\end{figure}

\subsubsection{Failure Analysis of Flow-based Temporal Propagation}

Although localized diffusion can improve difficult regions on key frames, it also introduces temporal inconsistency because enhanced frames may become noticeably sharper than their neighbors.
To reduce this flickering effect, we further explored flow-based residual propagation, where high-quality details generated on key frames are propagated to nearby frames using optical flow.

However, this direction did not become our final solution.
Both our experiments and the underlying task structure suggest that flow-based propagation is fundamentally weak for object removal and background reconstruction.
For instance, while using traditional OpenCV Farneback flow on the \texttt{wildvideo} dataset successfully propagated textures and increased the hard artifact gain to \(0.0695\), the temporal instability ratio remained high at \(3.724\). More critically, on the highly dynamic \texttt{bmx-trees} dataset, or when switching to the high-precision RAFT model, the propagation broke down completely (hard gain plummeting back to \(0.000\)).
The RAFT failure specifically highlighted severe engineering integration constraints, including forward/backward mapping mismatches and tensor normalization conflicts.

The fundamental reason is that this task is not merely about transporting visible pixels across time.
Instead, it often requires generating previously occluded background content that is never visible in adjacent frames.
Optical flow is effective for estimating correspondence between already visible appearances, but it is not designed to hallucinate missing content.

This mismatch becomes most severe near occlusion boundaries.
After object removal, the missing regions often involve disocclusion, one-to-many correspondence, and invalid backward mappings, making the inverse problem ill-posed.
Moreover, 2D flow models are easily corrupted by parallax, non-rigid motion, motion blur, and interpolation around high-frequency boundaries, which explains why seam and outside-region metrics are particularly sensitive to propagation strength.
In addition, propagation is recursive by nature, so even small alignment errors accumulate over time and manifest as temporal flicker.

Our quantitative results are consistent with this theoretical analysis.

\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Effect of propagation matching and flow models on the gain--risk trade-off.}
\label{tab:flow_failure}
\begin{tabular}{lccccccc}
\toprule
Dataset & Method & Radius & Outside $\downarrow$ & Seam $\downarrow$ & Leakage $\downarrow$ & Hard Gain $\uparrow$ & Temporal Ratio $\downarrow$ \\
\midrule
\multirow{3}{*}{\texttt{wildvideo}} & No Propagation     & 0 & 0.002300 & 0.284100 & 0.008900 & 0.041900 & 3.720000 \\
                                    & OpenCV Farneback   & 5 & 0.002100 & 0.284000 & 0.008600 & \textbf{0.069500} & 3.724000 \\
                                    & RAFT Flow          & 5 & 0.003100 & 0.377000 & 0.012500 & 0.000000 & 3.907000 \\
\midrule
\multirow{2}{*}{\texttt{bmx-trees}} & OpenCV Farneback   & 5 & -        & 1.224000 & -        & 0.000000 & 0.998000 \\
                                    & RAFT Flow          & 5 & -        & 1.117000 & -        & 0.000000 & 0.984000 \\
\bottomrule
\end{tabular}
\end{table*}

Overall, our experiments indicate that the diffusion-enhanced branch can improve some difficult single-frame cases, especially after sparse targeting and improved alpha blending.
However, it does not consistently outperform the baseline in overall evaluation, mainly because temporal stability remains the dominant bottleneck.
Therefore, we keep the earlier pipeline as the default system and regard the diffusion branch as an exploratory extension with informative failure cases.
"""

start_marker = r"\subsection{Ablation Study}"
end_marker = r"\section{Discussion and Future Direction}"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    # the start_marker is "\subsection{Ablation Study}\n\n\subsubsection{Sparse Diffusion Targeting}" from my previous edit.
    # we just replace the whole section starting from where \subsubsection{Sparse Diffusion Targeting} should be.
    # Actually, let's just find the \subsection{Ablation Study} and replace everything up to the Discussion section.
    
    # We want to remove the sections experimental setup and ablation study that I incorrectly added
    start_replace_idx = content.find(r"\section{Experiments and Analysis}") + len(r"\section{Experiments and Analysis}") + 1
    new_content = content[:start_replace_idx] + "\n" + new_text + "\n" + content[end_idx:]
    with open('report_architecture.tex', 'w') as f:
        f.write(new_content)
    print("Done")
else:
    print("Failed")

